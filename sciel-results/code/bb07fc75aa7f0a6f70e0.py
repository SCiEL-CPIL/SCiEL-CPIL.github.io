import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta."""
    start_time = time.time()

    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    # Progressive beta schedule: start smooth, end sharp
    betas = [20, 40, 80, 150, 300, 500, 800]

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
        for beta_sharp in [1500.0, 3000.0]:
            elapsed = time.time() - start_time
            if elapsed > time_budget - 2:
                break
            try:
                def objective_sharp(x):
                    return _negative_soft_radius(x, beta_sharp)

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
                    x0 = result.x.copy()
            except Exception:
                pass

    return best_x.reshape(-1, 2), best_score


def _slsqp_epigraph(centers, time_budget=20.0):
    """SLSQP epigraph formulation: maximize t subject to t <= pairwise dist/2 and t <= boundary dist."""
    start_time = time.time()
    n = N
    # Variables: 2N center coords + 1 radius t
    # Objective: minimize -t
    # Constraints: for each pair i<j, dist(c_i, c_j)/2 - t >= 0
    #               for each i, (0.5 - |c_i,k|) - t >= 0 for k in {0,1}
    #               -0.5 <= c_i,k <= 0.5

    x0 = np.concatenate([centers.ravel(), [_true_score(centers) * 0.95]])
    best_x = x0.copy()
    best_score = _true_score(centers)

    # Build constraint list
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    n_boundary = 2 * n

    def objective(x):
        return -x[-1]  # minimize -t

    def make_pair_constraint(i, j):
        def constraint(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            d = np.sqrt((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2) / 2.0
            return d - x[-1]
        return constraint

    def make_boundary_constraint(i, k):
        def constraint(x):
            return (0.5 - abs(x[2*i+k])) - x[-1]
        return constraint

    # Build constraints list
    constraints = []
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        constraints.append({'type': 'ineq', 'fun': make_pair_constraint(i, j)})
    for i in range(n):
        for k in range(2):
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, k)})

    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.5)]

    # Run SLSQP with time limit
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 3:
        try:
            maxiter = int(min(3000, remaining * 100))
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
            )

            new_centers = result.x[:2*n].reshape(n, 2)
            new_t = result.x[-1]
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    # Second pass with tighter tolerance if time remains
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 3:
        try:
            maxiter = int(min(2000, remaining * 80))
            result = minimize(
                objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
            )

            new_centers = result.x[:2*n].reshape(n, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x[:2*n].reshape(n, 2), best_score


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


def _local_refine(centers, time_budget=20.0):
    """Local refinement focusing on the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

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

            for i in order[:20]:
                if time.time() - start_time > time_budget:
                    break

                # Try 8 directions + radial
                angles = np.linspace(0, 2*np.pi, 9)[:-1]
                for angle in angles:
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
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

    # 2. For each init, do force relax + L-BFGS-B + SLSQP + SA + local refine
    for init_idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:
            break

        # Force relax
        relaxed = _force_relax(init, n_steps=200)
        score_relaxed = _true_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = relaxed.copy()

        # L-BFGS-B optimization
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 25:
            t_lbfgs = min(remaining * 0.3, 30.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgs(relaxed, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        # SLSQP epigraph - the key improvement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 20 and best_centers is not None:
            t_slsqp = min(remaining * 0.25, 25.0)
            centers_slsqp, score_slsqp = _slsqp_epigraph(best_centers, time_budget=t_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_centers = centers_slsqp.copy()

        # Simulated annealing
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 15 and best_centers is not None:
            t_sa = min(remaining * 0.15, 10.0)
            centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()

        # Delaunay refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 10 and best_centers is not None:
            t_del = min(remaining * 0.15, 7.0)
            centers_del, score_del = _delaunay_refine(best_centers, time_budget=t_del)
            if score_del > best_score:
                best_score = score_del
                best_centers = centers_del.copy()

        # Local refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 6 and best_centers is not None:
            t_lr = min(remaining * 0.3, 12.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # 3. Final SLSQP epigraph polish on best solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 20 and best_centers is not None:
        t_final_slsqp = min(remaining * 0.4, 35.0)
        centers_final, score_final = _slsqp_epigraph(best_centers, time_budget=t_final_slsqp)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()

    # 4. Final L-BFGS-B high-beta polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 12 and best_centers is not None:
        t_final_lbfgs = min(remaining * 0.4, 20.0)
        centers_final_lbfgs, score_final_lbfgs = _optimize_lbfgs(best_centers, time_budget=t_final_lbfgs)
        if score_final_lbfgs > best_score:
            best_score = score_final_lbfgs
            best_centers = centers_final_lbfgs.copy()

    # 5. Final Delaunay + local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 6 and best_centers is not None:
        t_final_del = min(remaining * 0.3, 5.0)
        centers_final_del, score_final_del = _delaunay_refine(best_centers, time_budget=t_final_del)
        if score_final_del > best_score:
            best_score = score_final_del
            best_centers = centers_final_del.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 3 and best_centers is not None:
            t_final_lr = min(remaining * 0.7, 8.0)
            centers_final_lr, score_final_lr = _local_refine(best_centers, time_budget=t_final_lr)
            if score_final_lr > best_score:
                best_score = score_final_lr
                best_centers = centers_final_lr.copy()

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
