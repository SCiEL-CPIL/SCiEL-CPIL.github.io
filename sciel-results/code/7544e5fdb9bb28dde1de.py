"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform


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
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _energy_focused(params, temperature=0.02):
    """Focused energy: only considers pairs within a threshold of the current minimum."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    if N < 2:
        return -b_min
    dists = pdist(c, metric='euclidean')
    d_min = np.min(dists)
    T = max(temperature, 1e-8)
    threshold = d_min * 1.15 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = d_min
    b_threshold = b_min * 1.15 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, trying multiple spacings and offsets."""
    best_c = None
    best_r = 0
    
    # Try hexagonal packings with various spacings
    for d in np.linspace(0.054, 0.076, 45):
        for x_shift in [0.0, 0.002, -0.002]:
            points = []
            row_spacing = d * np.sqrt(3) / 2
            x_offset = d / 2
            y = -0.5 + d / 2
            row = 0
            while y <= 0.5 - d / 2:
                x = -0.5 + d / 2 + x_shift
                if row % 2 == 1:
                    x += x_offset
                while x <= 0.5 - d / 2:
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        points.append([x, y])
                    x += d
                y += row_spacing
                row += 1
            if len(points) >= N:
                points = np.array(points)
                c = points[:N].copy()
                r = _compute_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
    
    # Also try square grid with slight perturbation
    m = int(np.ceil(np.sqrt(N)))
    for m_try in [m, m+1, m-1]:
        if m_try < 14:
            continue
        ticks = (np.arange(m_try) + 0.5) / m_try - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) >= N:
            c = grid[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: uniform grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _get_active_pairs(c, r_target, margin=0.008):
    """Get pairs within 2*r_target + margin using KD-tree."""
    n = c.shape[0]
    tree = cKDTree(c)
    dist_limit = 2 * r_target + margin
    pairs = []
    for i in range(n):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.append((i, j))
    if pairs:
        return np.array(pairs, dtype=np.int32)
    return np.empty((0, 2), dtype=np.int32)


def _sparse_objective(params, beta, pairs_arr):
    """Sparse objective using only active pairs + boundary."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    bd_min = np.min(boundary_dists)
    npairs = len(pairs_arr)
    if npairs == 0:
        return -bd_min
    ci = c[pairs_arr[:, 0]]
    cj = c[pairs_arr[:, 1]]
    diffs = ci - cj
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    pair_half = dists / 2.0
    ph_min = np.min(pair_half)
    # Soft min over active pairs + boundary
    all_vals = np.concatenate([pair_half, boundary_dists.ravel()])
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    r_soft = v_min - log_sum / beta
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _sparse_objective_grad(params, beta, pairs_arr):
    """Gradient of sparse objective."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    npairs = len(pairs_arr)
    all_vals = np.empty(npairs + 2 * n)
    if npairs > 0:
        ci = c[pairs_arr[:, 0]]
        cj = c[pairs_arr[:, 1]]
        diffs = ci - cj
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        all_vals[:npairs] = dists / 2.0
    all_vals[npairs:] = boundary_dists.ravel()
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    exp_vals = np.exp(-beta * shifted)
    total = np.sum(exp_vals)
    weights = exp_vals / total  # softmax weights (sum to 1)
    # d(-r_soft)/d(all_vals) = -weights
    grad = np.zeros(2 * n)
    # Boundary contribution: d(boundary_dists)/d(c) = -sign(c)
    b_weights = weights[npairs:]
    signs = -np.sign(c)
    for i in range(n):
        grad[2*i] += b_weights[2*i] * signs[i, 0]
        grad[2*i+1] += b_weights[2*i+1] * signs[i, 1]
    # Pair contribution
    if npairs > 0:
        p_weights = weights[:npairs]
        ci = c[pairs_arr[:, 0]]
        cj = c[pairs_arr[:, 1]]
        diffs = ci - cj
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists_safe = np.where(dists < 1e-12, 1e-12, dists)
        unit_vecs = diffs / dists_safe[:, np.newaxis]
        # d(pair_half)/d(ci) = unit_vec / 2, d(pair_half)/d(cj) = -unit_vec / 2
        # grad contribution = -p_weights * d(pair_half)/d(c)
        for k in range(npairs):
            w = p_weights[k]
            i_idx = pairs_arr[k, 0]
            j_idx = pairs_arr[k, 1]
            grad[2*i_idx] -= w * 0.5 * unit_vecs[k, 0]
            grad[2*i_idx+1] -= w * 0.5 * unit_vecs[k, 1]
            grad[2*j_idx] += w * 0.5 * unit_vecs[k, 0]
            grad[2*j_idx+1] += w * 0.5 * unit_vecs[k, 1]
    return grad


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing and sparse pairs."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        # Build active pairs for current radius estimate
        r_est = max(best_r, 0.03)
        pairs_arr = _get_active_pairs(best_c, r_est, margin=0.01)
        # If too many pairs, fall back to full
        if len(pairs_arr) > 3000:
            try:
                def obj(params):
                    return _smooth_objective(params, beta)
                maxiter = 60 if beta < 100 else 100
                result = minimize(
                    obj,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
            except Exception:
                pass
        else:
            try:
                def obj(params):
                    return _sparse_objective(params, beta, pairs_arr)
                def obj_grad(params):
                    return _sparse_objective_grad(params, beta, pairs_arr)
                maxiter = 100 if beta < 100 else 200
                result = minimize(
                    obj,
                    best_c.flatten(),
                    jac=obj_grad,
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
            except Exception:
                pass
    return best_c, best_r


def _local_optimize_focused(c0, max_time=20.0, temps=None):
    """Local optimization using the focused energy function with sparse pairs."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if temps is None:
        temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 1:
            break
        # Build focused pairs
        r_est = max(_compute_radius(current_c), 0.03)
        pairs_arr = _get_active_pairs(current_c, r_est, margin=0.006)
        if len(pairs_arr) > 2000:
            # Fall back to full focused energy
            def obj(params, _T=T):
                return _energy_focused(params, temperature=_T)
            try:
                result = minimize(
                    obj,
                    current_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
                )
                c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new.copy()
                    best_r = r_new
                current_c = c_new
            except Exception:
                continue
        else:
            # Sparse focused objective
            def obj_sparse(params, _T=T, _pairs=pairs_arr):
                c = np.clip(params.reshape(-1, 2), -0.4999, 0.4999)
                boundary_dists = 0.5 - np.abs(c)
                b_min = np.min(boundary_dists)
                npairs = len(_pairs)
                if npairs == 0:
                    return -b_min
                ci = c[_pairs[:, 0]]
                cj = c[_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                d_min = np.min(dists)
                threshold = d_min * 1.15 + 1e-6
                mask = dists < threshold
                if np.any(mask):
                    d_sub = dists[mask]
                    d_min_sub = np.min(d_sub)
                    soft_min_pair = d_min_sub - _T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / _T)))
                else:
                    soft_min_pair = d_min
                b_threshold = b_min * 1.15 + 1e-6
                b_mask = boundary_dists < b_threshold
                if np.any(b_mask):
                    b_sub = boundary_dists[b_mask]
                    b_min_sub = np.min(b_sub)
                    soft_min_bound = b_min_sub - _T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / _T)))
                else:
                    soft_min_bound = b_min
                r_pair = soft_min_pair / 2.0
                r_bound = soft_min_bound
                r_vals = np.array([r_pair, r_bound])
                m = np.min(r_vals)
                soft_min_combined = m - _T * np.log(np.sum(np.exp(-(r_vals - m) / _T)))
                return -soft_min_combined
            try:
                result = minimize(
                    obj_sparse,
                    current_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 250, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
                )
                c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new.copy()
                    best_r = r_new
                current_c = c_new
            except Exception:
                continue
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """Simulated annealing with rigid-body cluster moves to escape local minima."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    T_start = 0.001
    T_end = 1e-8
    n_steps = 5000

    def _build_clusters(c, r_target):
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return list(clusters.values())

    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 500

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        if np.random.random() < 0.6 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """High-precision polish using SLSQP with active constraints and nearby boundary only."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _get_pairs_and_boundary(c, r_target, margin=0.004):
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        # Only include boundary constraints for circles near the boundary
        bd = 0.5 - np.abs(c)
        near_boundary = np.where(np.min(bd, axis=1) < r_target + margin)[0]
        pairs_arr = np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
        return pairs_arr, near_boundary

    step = 1.5e-5
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        pairs, near_bnd = _get_pairs_and_boundary(best_c, r_target, margin=0.004)
        npairs = len(pairs)
        nbnd = len(near_bnd)

        def constraints_func(x):
            c = x.reshape(-1, 2)
            if nbnd > 0:
                bd = 0.5 - np.abs(c[near_bnd]) - r_target
                bd_flat = bd.ravel()
            else:
                bd_flat = np.array([])
            if npairs == 0:
                return bd_flat
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd_flat, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * nbnd + npairs
            jac = np.zeros((total_constraints, 2 * n))
            if nbnd > 0:
                signs = -np.sign(c[near_bnd])
                for k in range(nbnd):
                    i = near_bnd[k]
                    jac[2*k, 2*i] = signs[k, 0]
                    jac[2*k+1, 2*i+1] = signs[k, 1]
            if npairs == 0:
                return jac
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            row_indices = 2 * nbnd + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 300, 'ftol': 1e-15, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-14:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.08, 1.5e-4)
            else:
                step *= 0.5
                if step < 2e-8:
                    break
        except Exception:
            break
    return best_c, best_r


def _ultra_polish(c0, r0, max_time=10.0):
    """Ultra-precise final refinement using very low temperature focused energy with sparse pairs."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    temps = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 0.2:
            break
        r_est = max(_compute_radius(current_c), 0.03)
        pairs_arr = _get_active_pairs(current_c, r_est, margin=0.003)
        if len(pairs_arr) > 1500:
            def obj(params, _T=T):
                return _energy_focused(params, temperature=_T)
        else:
            def obj(params, _T=T, _pairs=pairs_arr):
                c = np.clip(params.reshape(-1, 2), -0.4999999, 0.4999999)
                boundary_dists = 0.5 - np.abs(c)
                b_min = np.min(boundary_dists)
                npairs = len(_pairs)
                if npairs == 0:
                    return -b_min
                ci = c[_pairs[:, 0]]
                cj = c[_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                d_min = np.min(dists)
                threshold = d_min * 1.1 + 1e-7
                mask = dists < threshold
                if np.any(mask):
                    d_sub = dists[mask]
                    d_min_sub = np.min(d_sub)
                    soft_min_pair = d_min_sub - _T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / _T)))
                else:
                    soft_min_pair = d_min
                b_threshold = b_min * 1.1 + 1e-7
                b_mask = boundary_dists < b_threshold
                if np.any(b_mask):
                    b_sub = boundary_dists[b_mask]
                    b_min_sub = np.min(b_sub)
                    soft_min_bound = b_min_sub - _T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / _T)))
                else:
                    soft_min_bound = b_min
                r_pair = soft_min_pair / 2.0
                r_bound = soft_min_bound
                r_vals = np.array([r_pair, r_bound])
                m = np.min(r_vals)
                soft_min_combined = m - _T * np.log(np.sum(np.exp(-(r_vals - m) / _T)))
                return -soft_min_combined
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999999, 0.4999999)] * (2 * N),
                options={'maxiter': 600, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 200}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999999, 0.4999999)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-15:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _randomize_clusters(c0, magnitude):
    """Randomize some clusters to create diversity."""
    n = c0.shape[0]
    r_est = _compute_radius(c0)
    tree = cKDTree(c0)
    dist_limit = 2.5 * r_est
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        neighbors = tree.query_ball_point(c0[i], dist_limit)
        for j in neighbors:
            if i < j:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    cluster_list = list(clusters.values())
    new_c = c0.copy()
    n_randomize = max(1, len(cluster_list) // 4)
    random_clusters = np.random.choice(len(cluster_list), n_randomize, replace=False)
    for ci in random_clusters:
        cluster = cluster_list[ci]
        angle = np.random.random() * 2 * np.pi
        mag = np.random.exponential(magnitude)
        dx = mag * np.cos(angle)
        dy = mag * np.sin(angle)
        for idx in cluster:
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
    return np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)


def construct():
    start_time = time.time()
    deadline = start_time + 149
    best_radius = 0.0
    best_c = None
    np.random.seed(42)

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Primary optimization with smooth objective (sparse)
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(30.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Focused energy optimization (sparse)
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(10.0, deadline - time.time() - 40)
        c_foc, r_foc = _local_optimize_focused(best_c, max_time=min(20.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()

    # 4. Simulated annealing with rigid-body moves
    if best_c is not None and time.time() < deadline - 30:
        remaining = max(10.0, deadline - time.time() - 30)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(20.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 5. Second smooth optimization pass
    if best_c is not None and time.time() < deadline - 20:
        remaining = max(5.0, deadline - time.time() - 20)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(12.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()

    # 6. Focused energy refinement after SA
    if best_c is not None and time.time() < deadline - 12:
        remaining = max(5.0, deadline - time.time() - 12)
        c_foc2, r_foc2 = _local_optimize_focused(best_c, max_time=min(10.0, remaining))
        if r_foc2 > best_radius:
            best_radius = r_foc2
            best_c = c_foc2.copy()

    # 7. Final high-precision SLSQP polish
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(12.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    # 8. Ultra-precise final refinement
    if best_c is not None and time.time() < deadline - 1.0:
        remaining = max(2.0, deadline - time.time() - 0.5)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(6.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra

    if best_c is None:
        best_c = _initialize_hexagonal()
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
