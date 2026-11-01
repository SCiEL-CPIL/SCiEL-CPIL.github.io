"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)

    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _get_active_pairs(c, k=12):
    """
    Get indices of active pairs (nearest neighbors) using KDTree.
    Returns a list of (i, j) tuples where i < j.
    """
    n = c.shape[0]
    tree = cKDTree(c)
    # Query k+1 to include self, so we get k neighbors
    dists, indices = tree.query(c, k=k + 1)

    pairs = set()
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))

    return list(pairs)


def _sparse_objective(params, beta, k_neighbors):
    """
    Objective function using sparse distances (K-NN only).
    Minimizes -soft_min(distances).
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]

    boundary_dists = 0.5 - np.abs(c)

    # Get active pairs
    pairs = _get_active_pairs(c, k_neighbors)

    if not pairs:
        # Fallback if no pairs found (should not happen)
        return -np.min(boundary_dists)

    # Compute distances for active pairs (vectorized)
    pi = np.array([p[0] for p in pairs], dtype=np.int64)
    pj = np.array([p[1] for p in pairs], dtype=np.int64)
    diffs = c[pi] - c[pj]
    pair_dists = np.sqrt(np.sum(diffs * diffs, axis=1))

    # Combine pair distances (halved) and boundary distances
    all_dists = np.concatenate([pair_dists / 2.0, boundary_dists.ravel()])

    r_soft = _soft_min(all_dists, beta)

    if r_soft <= 0:
        return 1.0
    return -r_soft


def _local_optimize_sparse(c0, max_time=30.0):
    """
    Local optimization using sparse objective with annealing.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    # Anneal beta from low to high
    betas = [10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    k_neighbors = 12  # Number of nearest neighbors to consider

    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break

        try:
            def obj(params):
                return _sparse_objective(params, beta, k_neighbors)

            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)

            if r_new > best_r:
                best_c = c_new
                best_r = r_new

        except Exception:
            pass

    return best_c, best_r


def _get_active_pairs_by_radius(c, r_target, margin=0.008):
    """Get active pairs (distance < 2*r_target + margin) using KDTree."""
    n = c.shape[0]
    tree = cKDTree(c)
    dist_limit = 2 * r_target + margin
    pairs = []
    for i in range(n):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def _slsqp_polish(c0, r0, max_time=40.0):
    """
    High-precision polish using SLSQP with vectorized active constraints.
    Iteratively raises r_target to push the packing tighter.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    # Adaptive step: start small, shrink if we fail
    step = 8e-5

    for _ in range(150):
        if time.time() > start_time + max_time:
            break

        r_target = best_r + step

        # Get active pairs for the target radius
        pairs = _get_active_pairs_by_radius(best_c, r_target, margin=0.008)
        npairs = len(pairs)

        def constraints_func(x):
            c = x.reshape(-1, 2)
            # Boundary constraints: 0.5 - |c_ik| - r >= 0
            bd = (0.5 - np.abs(c) - r_target).ravel()
            if npairs == 0:
                return bd
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total = 2 * n + npairs
            jac = np.zeros((total, 2 * n))

            # Boundary Jacobian: d(0.5-|c_ik|-r)/dc_ik = -sign(c_ik)
            signs = -np.sign(c)
            idx = np.arange(n)
            jac[2 * idx, 2 * idx] = signs[:, 0]
            jac[2 * idx + 1, 2 * idx + 1] = signs[:, 1]

            if npairs == 0:
                return jac

            # Pair Jacobian rows
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit = diffs / dists_safe[:, np.newaxis]

            base = 2 * n
            for k in range(npairs):
                i = pairs[k, 0]
                j = pairs[k, 1]
                r = base + k
                jac[r, 2 * i:2 * i + 2] = unit[k]
                jac[r, 2 * j:2 * j + 2] = -unit[k]

            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[{'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}],
                options={'maxiter': 60, 'ftol': 1e-12, 'disp': False}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)

            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                # Keep step roughly constant; it tracks the improvement
                step = max(4e-6, min(step, 0.5 * (r_new - (r_target - step))))
            else:
                # Shrink step and retry
                step *= 0.5
                if step < 2e-7:
                    break

        except Exception:
            step *= 0.5
            if step < 2e-7:
                break

    return best_c, best_r


def _final_lbfgs_polish(c0, max_time=15.0):
    """
    Cheap final precision pass: very tight L-BFGS-B on the smooth objective.
    Extracts additional precision from the local optimum.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    for beta in [800.0, 1500.0]:
        if time.time() > start_time + max_time - 2:
            break
        try:
            def obj(params):
                return _sparse_objective(params, beta, 14)

            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 4000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 80}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is not None:
        return best_c

    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)

    best_radius = 0.0
    best_c = None

    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # Primary optimization with sparse smooth objective
    if time.time() < deadline - 45:
        remaining = deadline - time.time() - 45
        c_opt, r_opt = _local_optimize_sparse(c_hex, max_time=min(45.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # Perturbed restarts (vectorized SLSQP polish is fast, so we can afford more)
    np.random.seed(42)
    n_trials = 6
    for trial in range(n_trials):
        if time.time() > deadline - 55:
            break

        if best_c is not None:
            noise_scale = 0.004 / (trial + 1)
            c_pert = best_c + np.random.normal(0, noise_scale, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))

        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()

        if time.time() < deadline - 50:
            remaining = deadline - time.time() - 50
            c_opt, r_opt = _local_optimize_sparse(c_pert, max_time=min(15.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()

    # High-precision SLSQP polish on the best solution found
    if best_c is not None and time.time() < deadline - 10:
        remaining = max(10.0, deadline - time.time() - 10)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    # Final tight-tolerance L-BFGS-B precision pass
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final, r_final = _final_lbfgs_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    if best_c is None:
        best_c = _initialize_hexagonal()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
# EVOLVE-BLOCK-END
