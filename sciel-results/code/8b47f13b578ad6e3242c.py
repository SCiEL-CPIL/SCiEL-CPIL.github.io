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


def _force_relax(centers, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20

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


def _exact_radius_negative(x):
    """Negative of the exact packing radius for SLSQP."""
    centers = x.reshape(-1, 2)
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return -min(boundary, min_dist / 2.0)


def _slsqp_epigraph_polish(centers, time_budget=15.0):
    """SLSQP epigraph formulation: maximize t s.t. all constraints >= t.
    
    Variables: [x_1..x_100, t] where t is the radius.
    Constraints:
      - boundary constraints: 0.5 - |x_i| >= t  for each i, each axis
      - pair constraints: |x_i - x_j|/2 >= t  for each pair
    Objective: minimize -t
    """
    start_time = time.time()
    centers = centers.copy()
    best_score = _true_score(centers)
    best_x = centers.ravel().copy()
    best_t = best_score

    def _epigraph_objective(v):
        return -v[-1]  # minimize -t

    def _epigraph_grad(v):
        g = np.zeros_like(v)
        g[-1] = -1.0
        return g

    # Build constraint functions
    # We'll use inequality constraints: g_k(v) >= 0
    # Boundary: for each point i and axis k: 0.5 - |v[2*i+k]| - v[-1] >= 0
    # Pair: for each pair (i,j): sqrt((v[2i]-v[2j])^2 + (v[2i+1]-v[2j+1])^2)/2 - v[-1] >= 0

    def _boundary_constraints(v):
        """Return array of boundary constraint values (all should be >= 0)."""
        t = v[-1]
        pts = v[:-1].reshape(-1, 2)
        # For each point and each axis
        cons = np.zeros(N * 2)
        for i in range(N):
            cons[2*i] = 0.5 - abs(pts[i, 0]) - t
            cons[2*i+1] = 0.5 - abs(pts[i, 1]) - t
        return cons

    def _pair_constraints(v):
        """Return array of pair constraint values (all should be >= 0)."""
        t = v[-1]
        pts = v[:-1].reshape(-1, 2)
        n_pairs = N*(N-1)//2
        cons = np.zeros(n_pairs)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                cons[idx] = d/2.0 - t
                idx += 1
        return cons

    # Try multiple SLSQP passes with increasing maxiter
    maxiters = [100, 200, 400, 800, 1500]

    x0 = np.concatenate([centers.ravel(), [best_t]])

    for maxiter in maxiters:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break

        try:
            # Use LinearConstraint for efficiency
            # Boundary constraints: A_b @ v >= 0
            # For 0.5 - |v_k| - t >= 0, this is non-linear due to abs.
            # We'll use non-linear constraint via 'ineq' in SLSQP
            
            # SLSQP with non-linear constraints
            def _all_ineq(v):
                """Combine all inequality constraints."""
                bc = _boundary_constraints(v)
                pc = _pair_constraints(v)
                return np.concatenate([bc, pc])

            result = minimize(
                _epigraph_objective,
                x0,
                method='SLSQP',
                jac=_epigraph_grad,
                bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                constraints=[
                    {'type': 'ineq', 'fun': _all_ineq}
                ],
                options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
            )

            new_t = result.x[-1]
            new_centers = np.clip(result.x[:-1].reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_x = result.x[:-1].copy()
                best_t = new_t
            
            # Update x0 for next iteration
            x0 = np.concatenate([best_x, [best_t]])
            
        except Exception as e:
            pass

    return best_x.reshape(-1, 2), best_score


def _optimize_slsqp(centers, time_budget=15.0):
    """Optimize using SLSQP with exact objective and hard constraints."""
    start_time = time.time()
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    bounds = [(-0.5, 0.5)] * (2 * N)

    # Multiple SLSQP passes with increasing maxiter
    maxiters = [100, 200, 400, 800]

    for maxiter in maxiters:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break

        try:
            result = minimize(
                _exact_radius_negative,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )

            new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta, then SLSQP polish."""
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

    # SLSQP polish on the best L-BFGS-B result
    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        t_slsqp = min(time_budget - elapsed - 1, 10.0)
        if t_slsqp > 2:
            try:
                centers_slsqp, score_slsqp = _optimize_slsqp(
                    best_x.reshape(-1, 2), time_budget=t_slsqp
                )
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = centers_slsqp.ravel().copy()
            except Exception:
                pass

    # Final high-precision SLSQP polish
    elapsed = time.time() - start_time
    if elapsed < time_budget - 2:
        try:
            result = minimize(
                _exact_radius_negative,
                best_x,
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _optimize_sa(centers, time_budget=10.0):
    """Simulated annealing refinement with adaptive step size and multi-point moves."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _true_score(centers)

    T = 0.005
    T_min = 0.00005
    cooling = 0.995
    step_scale = 0.3

    while time.time() - start_time < time_budget and T > T_min:
        n_move = 1
        if np.random.rand() < 0.2:
            n_move = 2
        elif np.random.rand() < 0.05:
            n_move = 3

        idxs = np.random.choice(N, size=n_move, replace=False)
        new_centers = centers.copy()
        for idx in idxs:
            delta = np.random.randn(2) * T * step_scale
            new_centers[idx] = np.clip(centers[idx] + delta, -0.5, 0.5)

        new_score = _true_score(new_centers)
        delta_score = new_score - best_score

        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()

        T *= cooling

    return best_centers, best_score


def _delaunay_refine(centers, time_budget=10.0):
    """Refine using Delaunay triangulation to identify and fix critical pairs."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break

        try:
            tri = Delaunay(best_centers)
            simplices = tri.simplices

            min_dist = np.inf
            min_pair = None
            for simplex in simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = np.linalg.norm(best_centers[simplex[i]] - best_centers[simplex[j]])
                        if d < min_dist:
                            min_dist = d
                            min_pair = (simplex[i], simplex[j])

            if min_pair is None:
                break

            i, j = min_pair
            direction = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                break
            direction = direction / norm

            for step in [0.001, 0.0005, 0.0002]:
                for sign in [1, -1]:
                    new_c = best_centers.copy()
                    new_c[i] += sign * direction * step
                    new_c[j] -= sign * direction * step
                    new_c = np.clip(new_c, -0.5, 0.5)
                    s = _true_score(new_c)
                    if s > best_score + 1e-12:
                        best_centers = new_c
                        best_score = s
                        break
        except Exception:
            break

    return best_centers, best_score


def _voronoi_style_refine(centers, time_budget=8.0):
    """Voronoi-style refinement: identify points with smallest Voronoi cell 
    (most constrained) and nudge them to equalize local density.
    
    Uses Delaunay triangulation to find nearest neighbors and applies
    force-based relaxation focused on the most constrained points.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    for iteration in range(30):
        if time.time() - start_time > time_budget:
            break

        try:
            tri = Delaunay(best_centers)
            vertices = tri.vertices  # (n_simplices, 3)
            
            # Build adjacency: for each point, find its neighbors
            neighbor_sets = [set() for _ in range(N)]
            for simplex in vertices:
                for i in range(3):
                    for j in range(i+1, 3):
                        neighbor_sets[simplex[i]].add(simplex[j])
                        neighbor_sets[simplex[j]].add(simplex[i])
            
            # Compute local "pressure" for each point: 
            # sum of (min_dist - actual_dist) for neighbors closer than min_dist
            diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # For each point, find its nearest neighbor distance
            nearest_dists = dists.min(axis=1)
            
            # Points with small nearest distance are most constrained
            # Apply outward force proportional to how constrained they are
            threshold = nearest_dists.min() * 1.5
            constrained_mask = nearest_dists < threshold
            
            forces = np.zeros_like(best_centers)
            for i in range(N):
                if not constrained_mask[i]:
                    continue
                # Push point away from its nearest neighbors
                for j in neighbor_sets[i]:
                    d = dists[i, j]
                    if d < 1e-10:
                        continue
                    # Force magnitude: stronger for closer neighbors
                    f_mag = max(0, threshold - d) / (d + 1e-10)
                    direction = (best_centers[i] - best_centers[j]) / d
                    forces[i] += f_mag * direction
            
            # Also apply boundary forces
            for k in range(2):
                mask_r = best_centers[:, k] > 0.45
                forces[mask_r, k] -= (best_centers[mask_r, k] - 0.45) * 10
                mask_l = best_centers[:, k] < -0.45
                forces[mask_l, k] += ((-0.45) - best_centers[mask_l, k]) * 10
            
            # Apply forces with small step
            step_size = 0.0005
            new_centers = best_centers + forces * step_size
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _true_score(new_centers)
            if new_score > best_score + 1e-12:
                best_centers = new_centers
                best_score = new_score
            else:
                # Try smaller step
                step_size = 0.0001
                new_centers = best_centers + forces * step_size
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_score = _true_score(new_centers)
                if new_score > best_score + 1e-12:
                    best_centers = new_centers
                    best_score = new_score
                else:
                    break  # No improvement, stop

        except Exception:
            break

    return best_centers, best_score


def _local_refine(centers, time_budget=20.0):
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

    # 1. Generate multiple initial configurations
    inits = []
    init_hex = _init_hexagonal()
    inits.append(init_hex)
    inits.append(_init_grid())
    inits.append((np.random.rand(N, 2) - 0.5) * 0.9)

    best_centers = None
    best_score = -1

    # 2. For each init, do force relax + L-BFGS-B + SA + local refine
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break

        # Force relax
        relaxed = _force_relax(init, n_steps=200)
        score_relaxed = _true_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = relaxed.copy()

        # L-BFGS-B + SLSQP optimization
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 20:
            t_lbfgs = min(remaining * 0.35, 35.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgs(relaxed, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        # Simulated annealing
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 12 and best_centers is not None:
            t_sa = min(remaining * 0.15, 10.0)
            centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()

        # Delaunay refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 8 and best_centers is not None:
            t_del = min(remaining * 0.15, 6.0)
            centers_del, score_del = _delaunay_refine(best_centers, time_budget=t_del)
            if score_del > best_score:
                best_score = score_del
                best_centers = centers_del.copy()

        # Local refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None:
            t_lr = min(remaining * 0.3, 12.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # 3. SLSQP epigraph polish on best solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15 and best_centers is not None:
        t_epigraph = min(remaining * 0.5, 30.0)
        centers_epigraph, score_epigraph = _slsqp_epigraph_polish(
            best_centers, time_budget=t_epigraph
        )
        if score_epigraph > best_score:
            best_score = score_epigraph
            best_centers = centers_epigraph.copy()

    # 3b. Standard SLSQP polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        t_final_slsqp = min(remaining * 0.4, 20.0)
        centers_final_slsqp, score_final_slsqp = _optimize_slsqp(
            best_centers, time_budget=t_final_slsqp
        )
        if score_final_slsqp > best_score:
            best_score = score_final_slsqp
            best_centers = centers_final_slsqp.copy()

    # 4. Final L-BFGS-B + SLSQP combined polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15 and best_centers is not None:
        t_final_lbfgs = min(remaining * 0.4, 25.0)
        centers_final, score_final = _optimize_lbfgs(best_centers, time_budget=t_final_lbfgs)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()

    # 5. Final Delaunay + Voronoi + local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        t_final_del = min(remaining * 0.25, 5.0)
        centers_final_del, score_final_del = _delaunay_refine(best_centers, time_budget=t_final_del)
        if score_final_del > best_score:
            best_score = score_final_del
            best_centers = centers_final_del.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 4 and best_centers is not None:
            t_final_vor = min(remaining * 0.3, 6.0)
            centers_final_vor, score_final_vor = _voronoi_style_refine(best_centers, time_budget=t_final_vor)
            if score_final_vor > best_score:
                best_score = score_final_vor
                best_centers = centers_final_vor.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 3 and best_centers is not None:
            t_final_lr = min(remaining * 0.5, 8.0)
            centers_final_lr, score_final_lr = _local_refine(best_centers, time_budget=t_final_lr)
            if score_final_lr > best_score:
                best_score = score_final_lr
                best_centers = centers_final_lr.copy()

    # 6. Final tight SLSQP epigraph polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        t_final_epi = min(remaining * 0.6, 12.0)
        centers_final_epi, score_final_epi = _slsqp_epigraph_polish(
            best_centers, time_budget=t_final_epi
        )
        if score_final_epi > best_score:
            best_score = score_final_epi
            best_centers = centers_final_epi.copy()

    # 7. Final exact SLSQP polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 3 and best_centers is not None:
        try:
            result = minimize(
                _exact_radius_negative,
                best_centers.ravel(),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass

    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers


def _optimize_lbfgs(centers, time_budget=15.0):
    """Alias for _optimize_lbfgsb."""
    return _optimize_lbfgsb(centers, time_budget=time_budget)


if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
