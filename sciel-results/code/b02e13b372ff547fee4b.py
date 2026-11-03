import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
DEADLINE = 140.0  # Safety margin before 150s limit


def _score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    d = pdist(centers)
    min_dist = d.min() if d.size > 0 else np.inf
    return min(boundary, min_dist / 2.0)


def _init_hexagonal():
    """Hexagonal initialization with proper spacing and selection."""
    best_pts = None
    best_r = -1

    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.49 and abs(y) <= 0.49:
                    pts.append([x, y])

        pts = np.array(pts)
        if len(pts) < N:
            continue

        # Select N points that maximize minimum distance (greedy farthest-point sampling)
        selected = [pts[np.argmin(np.sum(pts ** 2, axis=1))]]
        remaining = list(pts)
        selected[0] = pts[np.argmin(np.sum(pts ** 2, axis=1))]
        
        for _ in range(N - 1):
            sel = np.array(selected)
            if len(sel) > 0:
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            else:
                dists = np.sum(np.array(remaining) ** 2, axis=1)
            idx = np.argmax(dists)
            selected.append(remaining[idx])

        if len(selected) == N:
            pts_sel = np.array(selected)
            r = _score(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9


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

        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9


def _expand_radius(centers, r_start, r_target, time_budget=20.0):
    """
    Hard-disk radius expansion.
    Incrementally increase radius and resolve collisions.
    """
    start_time = time.time()
    centers = centers.copy()
    current_r = r_start
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])

    # Determine number of steps
    max_steps = 2000
    min_steps = 100
    steps = max(min_steps, min(max_steps, int(time_budget * 100)))

    for step in range(steps):
        if time.time() - start_time > time_budget:
            break

        # Target radius for this step
        target_r = r_start + (r_target - r_start) * (step + 1) / steps

        # Iterate to resolve collisions
        for _ in range(30):
            # Boundary constraint: |x| <= 0.5 - target_r
            limit = 0.5 - target_r
            centers = np.clip(centers, -limit, limit)

            # Pair constraints: |c_i - c_j| >= 2 * target_r
            diffs = centers[iu[0], :] - centers[iu[1], :]
            dists = np.sqrt((diffs ** 2).sum(axis=1))

            min_dist_req = 2 * target_r
            violations = dists < min_dist_req

            if not np.any(violations):
                break

            # Apply repulsive forces
            forces = np.zeros_like(centers)
            vi, vj = np.where(violations)
            for i, j in zip(vi, vj):
                d = dists[i, j]
                if d < 1e-12:
                    dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec)
                else:
                    dir_vec = diffs[i] / d

                overlap = min_dist_req - d
                forces[iu[0][i]] += 0.5 * overlap * dir_vec
                forces[iu[1][i]] -= 0.5 * overlap * dir_vec

            centers += forces
            centers = np.clip(centers, -limit, limit)

        current_r = target_r

    return centers


def _slsqp_optimize(centers, time_budget=15.0):
    """
    SLSQP epigraph optimization.
    Maximize t subject to:
    0.5 - |c_ik| >= t
    |c_i - c_j| >= 2t
    -0.5 <= c_ik <= 0.5
    """
    start_time = time.time()
    t0 = _score(centers)
    x0 = np.concatenate([centers.ravel(), [t0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])

    def objective(x):
        return -x[-1]

    def gradient(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        c = x[:2 * N].reshape(N, 2)
        t = x[-1]

        cons_bx = 0.5 - np.abs(c[:, 0]) - t
        cons_by = 0.5 - np.abs(c[:, 1]) - t

        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cons_p = dists - 2 * t

        return np.concatenate([cons_bx, cons_by, cons_p])

    try:
        # Determine maxiter based on time budget
        maxiter = max(50, min(200, int(time_budget * 10)))

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )

        if result.success:
            new_centers = result.x[:2 * N].reshape(N, 2)
            new_t = result.x[-1]
            true_t = _score(new_centers)
            if true_t >= t0 - 1e-9:
                return new_centers, true_t
    except Exception:
        pass

    return centers, _score(centers)


def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp."""
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    shifted = v - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum


def _optimize_lbfgsb(centers, time_budget=20.0):
    """L-BFGS-B with soft-min surrogate and progressive beta."""
    start_time = time.time()
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _score(centers)
    iu = np.triu_indices(N, k=1)

    betas = [20, 50, 100, 200, 400, 800]

    def objective(x, beta):
        c = x.reshape(N, 2)
        boundary_dists = 0.5 - np.abs(c)
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1)) / 2.0
        all_dists = np.concatenate([boundary_dists.ravel(), dists])
        return -_soft_min(all_dists, beta)

    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        try:
            result = minimize(
                lambda x: objective(x, beta),
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _local_refine(centers, time_budget=10.0):
    """Local refinement focusing on the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    iu = np.triu_indices(N, k=1)

    step_sizes = [0.002, 0.001, 0.0005, 0.0002, 0.0001]

    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break

        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False

            # Find most constrained points
            boundary_dists = (0.5 - np.abs(best_centers)).min(axis=1)
            pair_dists = np.zeros(N)
            for i in range(N):
                diffs = best_centers[i] - best_centers
                dists = np.sqrt((diffs ** 2).sum(axis=1))
                np.fill_diagonal(dists, np.inf)
                pair_dists[i] = dists.min() / 2.0

            constraints = np.minimum(boundary_dists, pair_dists)
            order = np.argsort(constraints)

            for i in order[:20]:
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
                        s = _score(new_c)
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

    # Generate initial configurations
    init_hex = _init_hexagonal()
    init_grid = _init_grid()

    best_centers = None
    best_score = -1

    # Process each initialization
    for init in [init_hex, init_grid]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break

        score_init = _score(init)
        if score_init > best_score:
            best_score = score_init
            best_centers = init.copy()

        # Phase 1: Hard-disk radius expansion
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 20 and best_centers is not None:
            t_expand = min(remaining * 0.4, 30.0)
            target_r = max(best_score + 0.003, 0.073)
            expanded = _expand_radius(best_centers, best_score, target_r, time_budget=t_expand)
            score_expanded = _score(expanded)
            if score_expanded > best_score:
                best_score = score_expanded
                best_centers = expanded.copy()

        # Phase 2: SLSQP polish
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 15 and best_centers is not None:
            t_slsqp = min(remaining * 0.5, 25.0)
            centers_slsqp, score_slsqp = _slsqp_optimize(best_centers, time_budget=t_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_centers = centers_slsqp.copy()

        # Phase 3: L-BFGS-B with soft-min surrogate
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 12 and best_centers is not None:
            t_lbfgs = min(remaining * 0.6, 20.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgsb(best_centers, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        # Phase 4: Local refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 8 and best_centers is not None:
            t_lr = min(remaining * 0.8, 15.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # Final polish if time remains
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        t_final = min(remaining * 0.5, 20.0)
        centers_final, score_final = _slsqp_optimize(best_centers, time_budget=t_final)
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
        best_centers = init_grid

    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers


if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(c)}")
    print(f"Shape: {c.shape}")
