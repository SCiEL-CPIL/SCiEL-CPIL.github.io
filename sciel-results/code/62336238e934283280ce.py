import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 50
DEADLINE = 148.0  # Safety margin before 150s kill

def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp for differentiability."""
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    shifted = v - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum

def _slsqp_epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -t subject to soft constraints.
    x = [centers (2N), t]
    Objective: -t + soft_penalty(constraint_violations)
    """
    centers = x[:2*N].reshape(-1, 2)
    t = x[2*N]
    
    # Boundary distances: 0.5 - |x_i|
    boundary_dists = 0.5 - np.abs(centers)
    
    # Pair distances / 2
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    
    # Violations: t - dist. We want dist >= t.
    violations = t - all_dists
    
    # Soft penalty: sum of softplus(violations)
    # softplus(x) = log(1 + exp(x))
    # We scale by beta to control sharpness
    soft_penalty = np.sum(np.logaddexp(0, beta * violations)) / beta
    
    return -t + soft_penalty

def _slsqp_epigraph_grad(x, beta=100.0):
    """
    Gradient of the epigraph objective.
    d/dx [ -t + sum_i log(1+exp(beta*(t-d_i)))/beta ]
    """
    centers = x[:2*N].reshape(-1, 2)
    t = x[2*N]
    
    grad = np.zeros_like(x)
    
    # Distances
    boundary_dists = 0.5 - np.abs(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    violations = t - all_dists
    
    # Derivative of softplus(beta * v) / beta is sigmoid(beta * v)
    # sigmoid(z) = 1 / (1 + exp(-z))
    sigmoids = 1.0 / (1.0 + np.exp(-beta * violations))
    
    # Gradient with respect to t
    # d/dt (-t) = -1
    # d/dt (sum penalty) = sum(sigmoid)
    grad[2*N] = -1.0 + np.sum(sigmoids)
    
    # Gradient with respect to centers
    # d/dx_i (penalty_i) = sigmoid_i * d/dx_i (t - d_i) = -sigmoid_i * d/dx_i (d_i)
    
    # Boundary part: d_i = 0.5 - |x_i,k|
    # d/dx_i,k (d_i) = -sign(x_i,k)
    # So grad_x += -sigmoid * (-sign) = sigmoid * sign
    b_sigmoids = sigmoids[:N]
    for i in range(N):
        for k in range(2):
            grad[i*2+k] += b_sigmoids[i] * np.sign(centers[i, k])
            
    # Pair part: d_ij = 0.5 * ||x_i - x_j||
    # d/dx_i (d_ij) = 0.5 * (x_i - x_j) / ||x_i - x_j||
    # d/dx_j (d_ij) = 0.5 * (x_j - x_i) / ||x_i - x_j||
    
    # We need to accumulate gradients for all pairs
    # sigmoids[N:] correspond to pairs in iu order
    p_sigmoids = sigmoids[N:]
    
    # To avoid O(N^2) python loops, we can use vectorized operations or sparse accumulation
    # But N=50 is small enough that a loop over pairs is fine if we are careful, 
    # or we can just let SLSQP use numerical gradients if this is too slow.
    # However, providing analytic gradient is much faster for SLSQP.
    
    # Let's try to vectorize the pair gradient accumulation.
    # This is tricky with the triu indices. 
    # Alternative: Use numerical gradient for SLSQP if analytic is complex.
    # Given N=50, numerical gradient (100 evals per step) might be okay if steps are few.
    # But analytic is better. Let's implement the loop.
    
    # Actually, let's just return None and let SLSQP use numerical gradients.
    # For N=50, 100 function evaluations per gradient step is acceptable if we limit iterations.
    # The function evaluation is O(N^2) which is 2500 ops, very fast.
    # 100 * 2500 = 250k ops per gradient. SLSQP might take 50-100 steps. 25M ops. Trivial.
    return None

def _optimize_epigraph(centers, time_budget=20.0):
    """
    Optimize using SLSQP with epigraph formulation.
    x = [centers (2N), t]
    """
    start_time = time.time()
    
    r_init = _true_score(centers)
    x0 = np.concatenate([centers.ravel(), [r_init]])
    best_x = x0.copy()
    best_score = r_init

    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    # Progressive beta: start smooth, increase sharpness
    beta_schedule = [20, 50, 100, 200, 400, 800]
    maxiters_schedule = [100, 200, 300, 400, 500, 600]

    for beta, maxiter in zip(beta_schedule, maxiters_schedule):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        try:
            def obj(x, b=beta):
                return _slsqp_epigraph_objective(x, beta=b)

            result = minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'iprint': 0}
            )

            new_centers = np.clip(result.x[:2*N].reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            
            # Update x0 for next iteration
            # Ensure t is consistent with the new centers
            x0[:2*N] = new_centers.ravel()
            x0[2*N] = new_score
            
        except Exception:
            pass

    return best_x[:2*N].reshape(-1, 2), best_score

def _voronoi_style_refine(centers, time_budget=10.0):
    """
    Voronoi-style refinement: push under-constrained circles toward neighbors.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break

        # Compute constraints
        diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists_sq = (diffs ** 2).sum(axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        boundary_dists = (0.5 - np.abs(best_centers)).min(axis=1)

        min_pair_dists = dists.min(axis=1) / 2.0
        circle_constraints = np.minimum(boundary_dists, min_pair_dists)
        r_current = circle_constraints.min()

        # Find under-constrained circles
        # Threshold slightly above min to identify "free" circles
        threshold = r_current + 0.0005
        under_constrained = np.where(circle_constraints > threshold)[0]

        if len(under_constrained) == 0:
            break

        improved = False
        # Shuffle to avoid bias
        np.random.shuffle(under_constrained)
        
        for idx in under_constrained:
            if time.time() - start_time > time_budget:
                break

            # Find nearest neighbor
            nearest_dist = dists[idx].copy()
            nearest_dist[idx] = np.inf
            nearest_idx = np.argmin(nearest_dist)

            direction = best_centers[nearest_idx] - best_centers[idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm

            # Try pushing toward neighbor
            # We want to increase the minimum distance, so moving toward the closest neighbor
            # might increase the distance to that neighbor if it's the binding constraint?
            # No, moving TOWARD the neighbor DECREASES the distance to it.
            # We should move AWAY from the closest neighbor if it's the binding constraint?
            # Or move towards the FARTHEST neighbor?
            
            # Actually, the standard Voronoi refinement logic:
            # If a circle is not on the boundary of the current radius, it has "room".
            # Moving it towards its nearest neighbor reduces the distance to that neighbor.
            # This seems counter-intuitive. 
            # Let's re-read the inspiration: "push under-constrained circles toward their nearest neighbors".
            # Why? Because if circle A is far from B, and B is close to C (binding), 
            # moving A towards B might allow B to move away from C? No, B is fixed in this step.
            
            # Alternative interpretation: 
            # The "Voronoi" cell of an under-constrained circle is large.
            # Moving it towards the center of its cell (which is roughly towards neighbors)
            # might help "compact" the packing, allowing the global minimum to rise
            # when other circles are adjusted in subsequent steps.
            
            # Let's try moving AWAY from the closest neighbor first, as that directly increases that pair's distance.
            # If that increases the global min, it's a win.
            
            # Let's try both directions: towards and away from nearest neighbor.
            for sign in [1, -1]: # 1: towards, -1: away
                for step in [0.001, 0.0005, 0.0002]:
                    new_c = best_centers.copy()
                    new_c[idx] = best_centers[idx] + sign * direction * step
                    new_c = np.clip(new_c, -0.5, 0.5)
                    s = _true_score(new_c)
                    if s > best_score + 1e-12:
                        best_centers = new_c
                        best_score = s
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            # Try a small random jitter for under-constrained circles if no improvement
            if np.random.rand() < 0.1:
                idx = np.random.choice(under_constrained)
                new_c = best_centers.copy()
                new_c[idx] += np.random.randn(2) * 0.001
                new_c = np.clip(new_c, -0.5, 0.5)
                s = _true_score(new_c)
                if s > best_score + 1e-12:
                    best_centers = new_c
                    best_score = s
                    improved = True
            if not improved:
                break

    return best_centers, best_score

def _init_hexagonal():
    """Improved hexagonal initialization."""
    best_pts = None
    best_r = -1

    # Hexagonal packing has density pi/sqrt(12) ~ 0.9069
    # For 50 circles in unit square, area = 50 * pi * r^2 <= 1
    # r <= 1/sqrt(50*pi) ~ 0.0797
    # Known best is ~0.0713.
    
    # Generate a large hexagonal grid
    s = 0.14 # spacing
    pts = []
    row_height = s * np.sqrt(3) / 2
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = j * s + (i % 2) * s / 2
            y = i * row_height
            if abs(x) <= 0.49 and abs(y) <= 0.49:
                pts.append([x, y])

    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()

    # Select N points that maximize the minimum distance (greedy)
    selected = []
    remaining = pts.copy()
    
    # Start with the point closest to center? Or furthest?
    # Center is usually good.
    center_dists = np.linalg.norm(remaining, axis=1)
    idx = np.argmin(center_dists)
    selected.append(remaining[idx])
    remaining = np.delete(remaining, idx, axis=0)

    for _ in range(N - 1):
        if len(remaining) == 0:
            break
        sel = np.array(selected)
        dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
        idx = np.argmax(dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)

    if len(selected) == N:
        pts_sel = np.array(selected)
        r = _true_score(pts_sel)
        if r > best_r:
            best_r = r
            best_pts = pts_sel

    if best_pts is not None:
        return best_pts
    return _init_grid()

def _init_grid():
    """Grid initialization."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7), (5, 5)]:
        total = rows * cols
        if total < N:
            continue

        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])

        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]

        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9

def _force_relax(centers, n_steps=100, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        # Boundary forces
        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20

        # Pair repulsion
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)

        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)

        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)

        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)
        lr *= lr_decay

    return centers

def _local_refine(centers, time_budget=5.0):
    """Local refinement focusing on the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002]

    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break

        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False

            boundary_dists = (0.5 - np.abs(best_centers)).min(axis=1)
            diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dists = dists.min(axis=1) / 2.0
            constraints = np.minimum(boundary_dists, min_pair_dists)

            order = np.argsort(constraints)

            for i in order[:10]:
                if time.time() - start_time > time_budget:
                    break

                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = best_centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                        s = _true_score(new_c)
                        if s > best_score + 1e-12:
                            best_centers = new_c
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_centers, best_score

def construct():
    """Main entry point."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Initialization
    init_hex = _init_hexagonal()
    init_grid = _init_grid()
    
    # Relax the hexagonal init to get a valid starting point
    relaxed_hex = _force_relax(init_hex, n_steps=100)
    relaxed_grid = _force_relax(init_grid, n_steps=100)
    
    best_centers = relaxed_hex
    best_score = _true_score(relaxed_hex)
    
    if _true_score(relaxed_grid) > best_score:
        best_centers = relaxed_grid
        best_score = _true_score(relaxed_grid)

    # 2. Main Optimization Loop
    # Alternate between Epigraph SLSQP and Voronoi Refinement
    # Epigraph is good for global moves, Voronoi for local tightening
    
    num_iterations = 10
    for i in range(num_iterations):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        
        # A. Epigraph Optimization
        t_epi = min(remaining * 0.6, 30.0)
        centers_epi, score_epi = _optimize_epigraph(best_centers, time_budget=t_epi)
        if score_epi > best_score:
            best_score = score_epi
            best_centers = centers_epi.copy()
        
        # B. Voronoi Refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
            break
        t_voro = min(remaining * 0.3, 10.0)
        centers_voro, score_voro = _voronoi_style_refine(best_centers, time_budget=t_voro)
        if score_voro > best_score:
            best_score = score_voro
            best_centers = centers_voro.copy()
            
        # C. Local Refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 3:
            break
        t_loc = min(remaining * 0.1, 5.0)
        centers_loc, score_loc = _local_refine(best_centers, time_budget=t_loc)
        if score_loc > best_score:
            best_score = score_loc
            best_centers = centers_loc.copy()

    # 3. Final Polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        # One more epigraph pass with high beta
        t_final = min(remaining * 0.8, 40.0)
        centers_final, score_final = _optimize_epigraph(best_centers, time_budget=t_final)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
            
        # Final Voronoi
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5:
            t_final_v = min(remaining * 0.5, 10.0)
            centers_fv, score_fv = _voronoi_style_refine(best_centers, time_budget=t_final_v)
            if score_fv > best_score:
                best_score = score_fv
                best_centers = centers_fv.copy()
                
    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
