"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
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
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    betas = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0]
    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            maxiter = 100 if beta < 50 else 180
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 80}
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
    """Local optimization using the focused energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if temps is None:
        temps = [0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
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
    T_start = 0.0015
    T_end = 1e-9
    n_steps = 8000

    def _build_clusters(c, r_target):
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.3 * r_target
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
    cluster_rebuild_interval = 400

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        # Choose move type: 55% cluster, 45% single
        if np.random.random() < 0.55 and len(clusters) > 1:
            # Pick a random cluster, weight by size for more interesting moves
            sizes = [len(cl) for cl in clusters]
            total = sum(sizes)
            probs = [s / total for s in sizes]
            cluster_idx = np.random.choice(len(clusters), p=probs)
            cluster = clusters[cluster_idx]
            
            # Random rotation + translation
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.004)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            # Apply rotation around cluster centroid
            centroid = np.mean(current_c[cluster], axis=0)
            cos_a, sin_a = np.cos(angle * 0.1), np.sin(angle * 0.1)  # Small rotation
            for idx in cluster:
                vx = current_c[idx, 0] - centroid[0]
                vy = current_c[idx, 1] - centroid[1]
                new_c[idx, 0] = centroid[0] + vx * cos_a - vy * sin_a + dx
                new_c[idx, 1] = centroid[1] + vx * sin_a + vy * cos_a + dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.0025)
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
    """High-precision polish using SLSQP with active constraints."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _compute_active_set(c, r_target, margin=0.004):
        """Compute which constraints are active (near their limit)."""
        bd = 0.5 - np.abs(c)
        active_bound = np.where(bd.ravel() < r_target + margin)[0]
        
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        active_pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    d = np.linalg.norm(c[i] - c[j])
                    if d < 2 * r_target + margin:
                        active_pairs.append((i, j))
        return active_bound, np.array(active_pairs, dtype=np.int32) if active_pairs else np.empty((0, 2), dtype=np.int32)

    step = 2.5e-5
    max_iter = 150
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        active_bound, active_pairs = _compute_active_set(best_c, r_target, margin=0.005)
        npairs = len(active_pairs)
        nbound = len(active_bound)
        
        if npairs == 0 and nbound == 0:
            # No active constraints, can increase radius directly
            best_r = r_target
            step = min(step * 1.1, 1e-4)
            continue

        def constraints_func(x):
            c = x.reshape(-1, 2)
            parts = []
            if nbound > 0:
                bd = 0.5 - np.abs(c)
                parts.append(bd.ravel()[active_bound] - r_target)
            if npairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                parts.append(dists - 2 * r_target)
            return np.concatenate(parts) if parts else np.zeros(0)

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = nbound + npairs
            if total_constraints == 0:
                return np.zeros((0, 2 * n))
            
            jac = np.zeros((total_constraints, 2 * n))
            row = 0
            if nbound > 0:
                # Boundary constraints: d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
                for idx in active_bound:
                    i = idx // 2
                    k = idx % 2
                    jac[row, 2*i + k] = -np.sign(c[i, k])
                    row += 1
            
            if npairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                # Vectorized accumulation using np.add.at
                row_offset = row
                for p in range(npairs):
                    i, j = active_pairs[p]
                    jac[row_offset + p, 2*i] += unit_vecs[p, 0]
                    jac[row_offset + p, 2*i + 1] += unit_vecs[p, 1]
                    jac[row_offset + p, 2*j] -= unit_vecs[p, 0]
                    jac[row_offset + p, 2*j + 1] -= unit_vecs[p, 1]
                row += npairs
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
                options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-13:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.1, 2e-4)
            else:
                step *= 0.5
                if step < 3e-8:
                    break
        except Exception:
            break
    return best_c, best_r


def _ultra_polish(c0, r0, max_time=12.0):
    """Ultra-precise final refinement using very low temperature focused energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    temps = [0.0003, 0.00015, 0.00008, 0.00004, 0.00002, 0.00001, 0.000005, 0.000002]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 0.5:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-17, 'gtol': 1e-13, 'maxls': 150}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _spectral_objective(params, alpha=50.0):
    """
    Spectral objective: maximize the smallest eigenvalue of a Laplacian-like
    matrix constructed from pairwise distance constraints.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    dists = pdist(c, metric='euclidean')
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    
    eps = 1e-8
    dists_matrix = squareform(dists)
    np.fill_diagonal(dists_matrix, 1.0)
    W = 1.0 / (dists_matrix + eps)
    np.fill_diagonal(W, 0.0)
    
    boundary_penalty = np.sum(np.maximum(0, 0.001 - boundary_dists)) ** 2
    
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    L += eps * np.eye(n)
    
    try:
        eigvals = np.linalg.eigvalsh(L)
        non_zero_eigs = eigvals[eigvals > 1e-6]
        if len(non_zero_eigs) > 0:
            lambda_min = non_zero_eigs[0]
        else:
            lambda_min = 0.0
    except np.linalg.LinAlgError:
        lambda_min = 0.0
    
    min_dist = np.min(dists)
    
    obj = -(alpha * lambda_min + min_dist / 2.0 + b_min)
    
    return obj + boundary_penalty


def _spectral_optimize(c0, max_time=25.0, alphas=None):
    """Local optimization using spectral objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if alphas is None:
        alphas = [10.0, 30.0, 50.0, 100.0]
    current_c = c0.copy()
    for alpha in alphas:
        if time.time() > start_time + max_time - 2:
            break
        def obj(params, _alpha=alpha):
            return _spectral_objective(params, alpha=_alpha)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 60}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
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
    return np.clip(new_c, -0.5 + 1e-12, 0.5 + 1e-12)


def _fast_slsqp_polish(c0, r0, max_time=15.0):
    """
    Faster SLSQP polish using a vectorized constraint Jacobian and
    a tighter active-set margin to reduce constraint count.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _compute_active_set_fast(c, r_target, margin=0.003):
        bd = 0.5 - np.abs(c)
        active_bound = np.where(bd.ravel() < r_target + margin)[0]
        
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        active_pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    d = np.linalg.norm(c[i] - c[j])
                    if d < 2 * r_target + margin:
                        active_pairs.append((i, j))
        if active_pairs:
            return active_bound, np.array(active_pairs, dtype=np.int32)
        return active_bound, np.empty((0, 2), dtype=np.int32)

    step = 3e-5
    max_iter = 200
    for it in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        active_bound, active_pairs = _compute_active_set_fast(best_c, r_target, margin=0.004)
        npairs = len(active_pairs)
        nbound = len(active_bound)
        
        if npairs == 0 and nbound == 0:
            best_r = r_target
            step = min(step * 1.15, 1.5e-4)
            continue

        def constraints_func(x):
            c = x.reshape(-1, 2)
            parts = []
            if nbound > 0:
                bd = 0.5 - np.abs(c)
                parts.append(bd.ravel()[active_bound] - r_target)
            if npairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                parts.append(dists - 2 * r_target)
            return np.concatenate(parts) if parts else np.zeros(0)

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = nbound + npairs
            if total_constraints == 0:
                return np.zeros((0, 2 * n))
            
            jac = np.zeros((total_constraints, 2 * n))
            row = 0
            if nbound > 0:
                for idx in active_bound:
                    i = idx // 2
                    k = idx % 2
                    jac[row, 2*i + k] = -np.sign(c[i, k])
                    row += 1
            
            if npairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                row_offset = row
                for p in range(npairs):
                    i, j = active_pairs[p]
                    jac[row_offset + p, 2*i] += unit_vecs[p, 0]
                    jac[row_offset + p, 2*i + 1] += unit_vecs[p, 1]
                    jac[row_offset + p, 2*j] -= unit_vecs[p, 0]
                    jac[row_offset + p, 2*j + 1] -= unit_vecs[p, 1]
                row += npairs
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
                options={'maxiter': 300, 'ftol': 1e-16, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-14:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.12, 3e-4)
            else:
                step *= 0.4
                if step < 5e-9:
                    break
        except Exception:
            break
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 148
    best_radius = 0.0
    best_c = None
    np.random.seed(42)

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(30.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Focused energy optimization
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(8.0, deadline - time.time() - 40)
        c_foc, r_foc = _local_optimize_focused(best_c, max_time=min(18.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()

    # 4. Spectral optimization to encourage uniform spreading
    if best_c is not None and time.time() < deadline - 35:
        remaining = max(8.0, deadline - time.time() - 35)
        c_spec, r_spec = _spectral_optimize(best_c, max_time=min(15.0, remaining))
        if r_spec > best_radius:
            best_radius = r_spec
            best_c = c_spec.copy()

    # 5. Simulated annealing with rigid-body moves
    if best_c is not None and time.time() < deadline - 25:
        remaining = max(10.0, deadline - time.time() - 25)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(22.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 6. Another smooth optimization pass
    if best_c is not None and time.time() < deadline - 18:
        remaining = max(5.0, deadline - time.time() - 18)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(12.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()

    # 7. Focused energy refinement after SA
    if best_c is not None and time.time() < deadline - 10:
        remaining = max(5.0, deadline - time.time() - 10)
        c_foc2, r_foc2 = _local_optimize_focused(best_c, max_time=min(8.0, remaining))
        if r_foc2 > best_radius:
            best_radius = r_foc2
            best_c = c_foc2.copy()

    # 8. Fast SLSQP polish
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final, r_final = _fast_slsqp_polish(best_c, best_radius, max_time=min(12.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    # 9. Ultra-precise final refinement
    if best_c is not None and time.time() < deadline - 0.5:
        remaining = max(1.5, deadline - time.time() - 0.3)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(8.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra

    if best_c is None:
        best_c = _initialize_hexagonal()
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
