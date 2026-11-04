"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

DEADLINE = 145.0  # Safety margin before 150s limit


def _contact_graph_refine(centers, r, n_iters=200, time_budget=15.0):
    """Refine using contact graph: optimize only binding constraints."""
    start_time = time.time()
    n = N
    centers = centers.copy()

    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break

        # Compute all distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = (diffs ** 2).sum(axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)

        boundary = (0.5 - np.abs(centers)).min()
        min_dist = dists.min()
        r_current = min(boundary, min_dist / 2.0)

        if r_current <= r:
            break
        r = r_current

        # Identify binding constraints (within 1% of min)
        threshold = r_current * 2.0 * 1.01
        binding_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < threshold:
                    binding_pairs.append((i, j))

        # Boundary constraints
        binding_boundary = []
        for i in range(n):
            for k in range(2):
                if 0.5 - np.abs(centers[i, k]) < r_current * 1.01:
                    binding_boundary.append((i, k))

        if not binding_pairs and not binding_boundary:
            break

        # Build gradient for binding constraints only
        grad = np.zeros_like(centers)

        for i, j in binding_pairs:
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            # Maximize min dist: push apart
            force = d / dist
            grad[i] += force
            grad[j] -= force

        for i, k in binding_boundary:
            sign = 1.0 if centers[i, k] > 0 else -1.0
            grad[i, k] -= sign  # Push inward

        # Normalize and take step
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            break
        step_size = 0.01 * r_current
        centers += step_size * grad / norm
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)

    final_r = _true_score(centers)
    if final_r > r:
        return centers, final_r
    return centers, r


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

            result = opt.minimize(
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

                result = opt.minimize(
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


def _slsqp_epigraph(centers, time_budget=20.0, warm_start_r=None):
    """SLSQP epigraph formulation: maximize t subject to t <= pairwise dist/2 and t <= boundary dist."""
    start_time = time.time()
    n = N

    def get_centers(x):
        return x[:2 * n].reshape(n, 2)

    def get_r(x):
        return x[-1]

    def objective(x):
        return -x[-1]

    def combined_constraints(x):
        c = get_centers(x)
        r = get_r(x)

        # Pairwise: dist^2 - (2r)^2 >= 0
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        pair_vals = dist_sq[mask] - 4 * r * r

        # Boundary: (0.5 - r)^2 - c_i^2 >= 0
        bound_x = (0.5 - r) ** 2 - c[:, 0] ** 2
        bound_y = (0.5 - r) ** 2 - c[:, 1] ** 2

        all_vals = np.concatenate([pair_vals, bound_x, bound_y])
        return all_vals

    constraints = [{'type': 'ineq', 'fun': combined_constraints}]

    # Estimate initial r with warm start
    if warm_start_r is not None and warm_start_r > 0:
        r_init = warm_start_r * 0.999
    else:
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = (diffs ** 2).sum(axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        d_min = np.sqrt(dists_sq.min())
        b_min = (0.5 - np.abs(centers)).min()
        r_init = min(d_min / 2, b_min) * 0.95

    x0 = np.concatenate([centers.flatten(), [r_init]])

    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.001, 0.5)]

    best_centers = centers.copy()
    best_score = _true_score(centers)

    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-12})

        new_centers = res.x[:2 * n].reshape(n, 2)
        new_score = _true_score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
    except Exception:
        pass

    # Second pass with tighter tolerance if time allows
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.6 and best_score > 0:
        try:
            x0_2 = np.concatenate([best_centers.flatten(), [best_score * 0.999]])
            res2 = opt.minimize(objective, x0_2, method='SLSQP', bounds=bounds, constraints=constraints,
                                options={'maxiter': 1500, 'ftol': 1e-14})

            new_centers2 = res2.x[:2 * n].reshape(n, 2)
            new_score2 = _true_score(new_centers2)
            if new_score2 > best_score:
                best_score = new_score2
                best_centers = new_centers2.copy()
        except Exception:
            pass

    return best_centers, best_score


def construct():
    start_time = time.time()

    # Generate multiple initial guesses
    guesses = []
    grid_pts = _init_grid()
    guesses.append(grid_pts)

    hex_pts = _init_hexagonal()
    guesses.append(hex_pts)

    # Add some random perturbations of the grid
    np.random.seed(42)
    for _ in range(3):
        perturbed = grid_pts + np.random.normal(0, 0.02, (N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        guesses.append(perturbed)

    best_centers = None
    best_r = 0.0

    for guess in guesses:
        if time.time() - start_time > DEADLINE - 30:
            break

        # Force relax to ensure valid starting point
        relaxed = _force_relax(guess, n_steps=200)
        relaxed_score = _true_score(relaxed)

        if relaxed_score > best_r:
            best_r = relaxed_score
            best_centers = relaxed.copy()

        # L-BFGS-B optimization with time budget
        remaining = DEADLINE - (time.time() - start_time) - 25
        if remaining < 5:
            break

        time_budget = min(remaining * 0.4, 20.0)
        lbfgsb_centers, lbfgsb_score = _optimize_lbfgsb(relaxed, time_budget=time_budget)

        if lbfgsb_score > best_r:
            best_r = lbfgsb_score
            best_centers = lbfgsb_centers.copy()

        # SLSQP epigraph optimization with warm start
        remaining = DEADLINE - (time.time() - start_time) - 15
        if remaining < 5:
            break

        slsqp_time_budget = min(remaining * 0.5, 25.0)
        slsqp_centers, slsqp_score = _slsqp_epigraph(lbfgsb_centers, time_budget=slsqp_time_budget,
                                                     warm_start_r=best_r)

        if slsqp_score > best_r:
            best_r = slsqp_score
            best_centers = slsqp_centers.copy()

        # Contact graph refinement
        remaining = DEADLINE - (time.time() - start_time) - 10
        if remaining > 5 and best_centers is not None:
            cg_time = min(remaining * 0.5, 10.0)
            cg_centers, cg_score = _contact_graph_refine(best_centers, best_r, time_budget=cg_time)
            if cg_score > best_r:
                best_r = cg_score
                best_centers = cg_centers.copy()

    # Final polish if we have time
    if best_centers is not None:
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining > 10:
            # Final SLSQP with warm start
            slsqp_time = min(remaining * 0.6, 20.0)
            final_slsqp, final_slsqp_score = _slsqp_epigraph(best_centers, time_budget=slsqp_time,
                                                             warm_start_r=best_r)
            if final_slsqp_score > best_r:
                best_r = final_slsqp_score
                best_centers = final_slsqp.copy()

            # Contact graph refinement on final result
            remaining2 = DEADLINE - (time.time() - start_time) - 3
            if remaining2 > 3:
                cg_final, cg_final_score = _contact_graph_refine(best_centers, best_r, time_budget=min(remaining2, 8.0))
                if cg_final_score > best_r:
                    best_r = cg_final_score
                    best_centers = cg_final.copy()

    # Fallback if nothing found
    if best_centers is None:
        best_centers = grid_pts

    return best_centers

# EVOLVE-BLOCK-END
