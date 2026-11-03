import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.optimize import LinearConstraint

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit


def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
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


def _soft_radius(centers, beta=50.0):
    """Differentiable soft approximation of the packing radius."""
    boundary_dists = 0.5 - np.abs(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    return _soft_min(all_dists, beta)


def _negative_soft_radius(x, beta=50.0):
    """Negative soft radius for minimization."""
    centers = x.reshape(-1, 2)
    return -_soft_radius(centers, beta)


def _init_grid():
    """Grid initialization trying multiple configurations."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
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


def _init_hexagonal():
    """Improved hexagonal initialization with proper spacing."""
    best_pts = None
    best_r = -1

    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    pts.append([x, y])

        pts = np.array(pts)
        if len(pts) < N:
            continue

        selected = []
        remaining = pts.copy()
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


def _expand_radius(centers, r_start, r_end, n_steps=1000):
    """
    Hard-disk radius expansion.
    Incrementally increase radius from r_start to r_end.
    At each step, resolve collisions by pushing points apart.
    """
    centers = centers.copy()
    current_r = r_start
    
    # Precompute indices for pairs
    iu = np.triu_indices(N, k=1)
    
    for step in range(n_steps):
        target_r = r_start + (r_end - r_start) * (step + 1) / n_steps
        
        # Check if we need to resolve collisions for this target radius
        # We want min_pair_dist >= 2 * target_r and boundary_dist >= target_r
        
        # Iterative collision resolution
        for _ in range(20):  # Max iterations per radius step
            # Check boundary
            # If a point is too close to boundary, push it in
            # Check pairs
            # If a pair is too close, push them apart
            
            # Vectorized check is faster, but let's do a simple loop for clarity and control
            # Actually, vectorized is better.
            
            # 1. Boundary constraints
            # dist_to_boundary = 0.5 - abs(c)
            # If dist_to_boundary < target_r, push towards center
            abs_c = np.abs(centers)
            dist_b = 0.5 - abs_c
            # For each coordinate, if dist_b < target_r, we need to move.
            # Simple projection:
            # x_new = sign(x) * max(abs(x), target_r) ? No, that's for box constraint.
            # We want 0.5 - |x| >= r => |x| <= 0.5 - r
            # So clip x to [-0.5+r, 0.5-r]
            limit = 0.5 - target_r
            centers = np.clip(centers, -limit, limit)
            
            # 2. Pair constraints
            # dist(c_i, c_j) >= 2 * target_r
            # Compute distances
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = (diffs ** 2).sum(axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            # Find violations
            min_dist_req = 2 * target_r
            violations = dists < min_dist_req
            np.fill_diagonal(violations, False)
            
            if not np.any(violations):
                break
                
            # Apply repulsive forces for violating pairs
            # Force magnitude proportional to overlap
            # F = (d_req - d) / d * direction
            # Direction is (c_i - c_j) / d
            # We want to push i away from j and j away from i
            
            # Accumulate forces
            forces = np.zeros_like(centers)
            
            # Only process violating pairs
            vi, vj = np.where(violations)
            for i, j in zip(vi, vj):
                d = dists[i, j]
                if d < 1e-10:
                    # Random nudge
                    d = 1e-10
                    dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec)
                else:
                    dir_vec = (centers[i] - centers[j]) / d
                
                overlap = min_dist_req - d
                # Force on i is +overlap * dir_vec
                # Force on j is -overlap * dir_vec
                # Scale factor to avoid explosion
                scale = 0.5 
                forces[i] += scale * overlap * dir_vec
                forces[j] -= scale * overlap * dir_vec
            
            centers += forces
            
            # Re-clip to boundary
            centers = np.clip(centers, -limit, limit)
            
        # Update current_r
        current_r = target_r
        
    return centers


def _slsqp_polish(centers, time_budget=10.0):
    """
    SLSQP polish using epigraph formulation.
    Maximize t subject to:
    0.5 - |c_ik| >= t for all i, k
    |c_i - c_j| >= 2t for all i < j
    -0.5 <= c_ik <= 0.5
    
    Minimize -t.
    """
    start_time = time.time()
    
    # Initial guess for t
    t0 = _true_score(centers)
    
    x0 = np.concatenate([centers.ravel(), [t0]])
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Constraints
    # We need to define constraints g(x) >= 0
    
    def objective(x):
        return -x[-1]
    
    # Gradient of objective
    def gradient(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
        
    # Constraints
    # 1. Boundary: 0.5 - |c_ik| - t >= 0
    # 2. Pairs: |c_i - c_j| - 2t >= 0
    
    # This is hard to vectorize easily for SLSQP's constraint function which expects 1D array.
    # We can use a list of constraint dicts or a single function returning array.
    
    # Let's use a single function returning array of all constraints.
    # Number of constraints: 2*N (boundary) + N*(N-1)/2 (pairs)
    n_pairs = N * (N - 1) // 2
    n_cons = 2 * N + n_pairs
    
    iu = np.triu_indices(N, k=1)
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        
        # Boundary constraints
        # 0.5 - |c_x| - t >= 0
        # 0.5 - |c_y| - t >= 0
        cons_bx = 0.5 - np.abs(c[:, 0]) - t
        cons_by = 0.5 - np.abs(c[:, 1]) - t
        
        # Pair constraints
        # |c_i - c_j| - 2t >= 0
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cons_p = dists - 2 * t
        
        return np.concatenate([cons_bx, cons_by, cons_p])
        
    # Jacobian of constraints (approximate or analytical)
    # For SLSQP, providing Jacobian is optional but helpful.
    # Let's try without first, or provide a simple numerical one if needed.
    # SLSQP can estimate it.
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if result.success:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_t = result.x[-1]
            # Verify
            true_t = _true_score(new_centers)
            if true_t >= t0 - 1e-9:
                return new_centers, true_t
        else:
            # If failed, maybe try with fewer iterations or different options
            pass
            
    except Exception:
        pass
        
    return centers, _true_score(centers)


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta."""
    start_time = time.time()

    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    # Progressive beta schedule: start smooth, end sharp
    betas = [20, 40, 80, 150, 300, 500]

    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        def objective(x):
            return _negative_soft_radius(x, beta)

        try:
            maxiter = 500 if beta < 100 else 800
            ftol = 1e-12
            gtol = 1e-10

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish with very sharp beta
    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        try:
            def objective_sharp(x):
                return _negative_soft_radius(x, 1000.0)

            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _local_refine(centers, time_budget=10.0):
    """Local refinement focusing on the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]

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

            for i in order[:15]:
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
    """Main entry point combining multiple optimization strategies."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Generate initial configurations
    inits = []
    init_hex = _init_hexagonal()
    inits.append(init_hex)
    inits.append(_init_grid())
    
    best_centers = None
    best_score = -1

    # 2. For each init, do Hard-Disk Expansion + SLSQP + Local Refine
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break

        # Initial score
        score_init = _true_score(init)
        if score_init > best_score:
            best_score = score_init
            best_centers = init.copy()

        # Hard-Disk Radius Expansion
        # Start from current radius, try to expand to a slightly higher target
        # Target radius: best known is ~0.0714. Let's aim for 0.072 or current + 0.001
        target_r = max(score_init + 0.002, 0.072)
        
        # Expansion is slow, so we limit time
        t_expand = 10.0
        if time.time() - start_time + t_expand > DEADLINE - 10:
            t_expand = max(0, DEADLINE - 10 - (time.time() - start_time))
            
        if t_expand > 2:
            # Estimate steps based on time? No, fixed steps but we check time inside?
            # _expand_radius doesn't check time. Let's wrap it or assume it's fast enough for few steps.
            # Actually, _expand_radius with 1000 steps and 20 iterations each is 20k distance calcs.
            # 50 points -> 1225 pairs. 20k * 1225 ~ 24M ops. Should be < 1s in NumPy?
            # Let's try 500 steps.
            expanded = _expand_radius(init, score_init, target_r, n_steps=500)
            score_expanded = _true_score(expanded)
            if score_expanded > best_score:
                best_score = score_expanded
                best_centers = expanded.copy()
            else:
                # If expansion failed to improve, maybe it got stuck. 
                # Use the expanded config anyway for SLSQP start?
                # Or stick to init?
                # Let's use the one with higher score.
                pass

        # SLSQP Polish
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 15 and best_centers is not None:
            t_slsqp = min(remaining * 0.5, 20.0)
            centers_slsqp, score_slsqp = _slsqp_polish(best_centers, time_budget=t_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_centers = centers_slsqp.copy()

        # L-BFGS-B Polish (as backup or further refinement)
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 10 and best_centers is not None:
            t_lbfgs = min(remaining * 0.4, 15.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgsb(best_centers, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        # Local Refine
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None:
            t_lr = min(remaining * 0.5, 10.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # 3. Final Global SLSQP + Local Refine
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        t_final_slsqp = min(remaining * 0.5, 15.0)
        centers_final, score_final = _slsqp_polish(best_centers, time_budget=t_final_slsqp)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()

    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        t_final_lr = min(remaining * 0.8, 10.0)
        centers_final_lr, score_final_lr = _local_refine(best_centers, time_budget=t_final_lr)
        if score_final_lr > best_score:
            best_score = score_final_lr
            best_centers = centers_final_lr.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers


if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
