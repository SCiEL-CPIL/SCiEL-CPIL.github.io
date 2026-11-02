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
    """Smooth approximation of min using log-sum-exp with tighter window."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Tighter window focuses on truly critical pairs
    shifted = np.clip(shifted, 0, 50.0 / beta)
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
    
    # More temperature levels for finer convergence
    betas = [15.0, 25.0, 40.0, 60.0, 100.0, 180.0, 300.0, 500.0, 800.0, 1200.0, 1800.0, 2500.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            maxiter = 60 if beta < 50 else (100 if beta < 300 else 150)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """
    Simulated annealing with rigid-body cluster moves (translation + rotation)
    to escape local minima while preserving local lattice geometry.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    
    # Temperature schedule
    T_start = 0.0012
    T_end = 1e-10
    n_steps = 10000
    
    def _build_clusters(c, r_target):
        """Group circles into small clusters (size 2-6) based on proximity."""
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.1 * r_target
        
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
        
        # Only keep clusters of size 2-6 for rigid body moves
        result = []
        for cl in clusters.values():
            if 2 <= len(cl) <= 6:
                result.append(cl)
            elif len(cl) > 6:
                # Split large clusters into smaller groups
                for k in range(0, len(cl), 4):
                    sub = cl[k:k+4]
                    if len(sub) >= 2:
                        result.append(sub)
        
        return result if result else [list(range(n))]
    
    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 600
    
    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        new_c = current_c.copy()
        move_type = np.random.random()
        
        if move_type < 0.50 and len(clusters) > 1:
            # Rigid body move: translation + small rotation
            cluster = clusters[np.random.randint(len(clusters))]
            n_cluster = len(cluster)
            
            # Compute centroid
            centroid = current_c[cluster].mean(axis=0)
            
            # Translation
            t_angle = np.random.random() * 2 * np.pi
            t_mag = np.random.exponential(0.002 / np.sqrt(n_cluster))
            dx = t_mag * np.cos(t_angle)
            dy = t_mag * np.sin(t_angle)
            
            # Small rotation around centroid
            rot_angle = np.random.exponential(0.02 / np.sqrt(n_cluster))
            if np.random.random() < 0.5:
                rot_angle = -rot_angle
            cos_r = np.cos(rot_angle)
            sin_r = np.sin(rot_angle)
            
            for idx in cluster:
                vx = current_c[idx, 0] - centroid[0]
                vy = current_c[idx, 1] - centroid[1]
                new_c[idx, 0] = centroid[0] + dx + cos_r * vx - sin_r * vy
                new_c[idx, 1] = centroid[1] + dy + sin_r * vx + cos_r * vy
                
        elif move_type < 0.75:
            # Single circle move with larger step
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.0012)
            new_c[idx, 0] += magnitude * np.cos(angle)
            new_c[idx, 1] += magnitude * np.sin(angle)
        else:
            # Swap two nearby circles
            i = np.random.randint(N)
            tree = cKDTree(current_c)
            _, j_candidates = tree.query(current_c[i], k=4)
            if isinstance(j_candidates, np.ndarray) and len(j_candidates) > 1:
                j = int(j_candidates[np.random.randint(1, len(j_candidates))])
                if j != i:
                    new_c[i], new_c[j] = new_c[j].copy(), new_c[i].copy()
        
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
    """
    High-precision polish using SLSQP with active constraints and
    iterative radius-increase with tighter convergence.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    def _get_active_pairs(c, r_target, margin=0.004):
        """Get pairs that are close to the target radius constraint."""
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
    
    def _get_boundary_indices(c, r_target, margin=0.008):
        """Get indices of circles near the boundary."""
        near_boundary = np.any(np.abs(c) > 0.5 - r_target - margin, axis=1)
        return np.where(near_boundary)[0]
    
    step = 2.5e-5
    max_iter = 150
    consecutive_failures = 0
    
    for iter_idx in range(max_iter):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + step
        margin = max(0.002, step * 1.5)
        
        pairs = _get_active_pairs(best_c, r_target, margin=margin)
        npairs = len(pairs)
        bd_indices = _get_boundary_indices(best_c, r_target)
        n_bd = len(bd_indices)
        
        total_constraints = 2 * n_bd + npairs
        if total_constraints == 0:
            # No active constraints, just push outward slightly
            break
        
        def constraints_func(x):
            c = x.reshape(-1, 2)
            parts = []
            if n_bd > 0:
                bd = 0.5 - np.abs(c[bd_indices]) - r_target
                parts.append(bd.ravel())
            if npairs > 0:
                ci = c[pairs[:, 0]]
                cj = c[pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                parts.append(dists - 2 * r_target)
            return np.concatenate(parts) if parts else np.zeros(1)
        
        def jac_func(x):
            c = x.reshape(-1, 2)
            jac = np.zeros((total_constraints, 2 * n))
            
            if n_bd > 0:
                for k, i in enumerate(bd_indices):
                    signs = -np.sign(c[i])
                    if abs(signs[0]) < 1e-10:
                        signs[0] = 1.0
                    if abs(signs[1]) < 1e-10:
                        signs[1] = 1.0
                    jac[2*k, 2*i] = signs[0]
                    jac[2*k+1, 2*i+1] = signs[1]
            
            if npairs > 0:
                ci = c[pairs[:, 0]]
                cj = c[pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                
                row_start = 2 * n_bd
                row_indices = row_start + np.arange(npairs)
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
                options={'maxiter': 250, 'ftol': 1e-16, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-14:
                best_c = c_new
                best_r = r_new
                consecutive_failures = 0
                step = min(step * 1.06, 1.2e-4)
            else:
                consecutive_failures += 1
                step *= 0.55
                if step < 3e-8 or consecutive_failures > 6:
                    break
                    
        except Exception:
            step *= 0.5
            consecutive_failures += 1
            if consecutive_failures > 4:
                break
            
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 147  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 70:
        remaining = deadline - time.time() - 70
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(42.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Simulated annealing with rigid-body moves (translation + rotation)
    if best_c is not None and time.time() < deadline - 50:
        remaining = max(15.0, deadline - time.time() - 50)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(38.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 4. Another smooth optimization pass from the SA result
    if best_c is not None and time.time() < deadline - 35:
        remaining = max(10.0, deadline - time.time() - 35)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(22.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # 5. Second short SA pass to escape any new local minima
    if best_c is not None and time.time() < deadline - 20:
        remaining = max(8.0, deadline - time.time() - 20)
        c_sa2, r_sa2 = _simulated_annealing_rigid(best_c, best_radius, max_time=min(15.0, remaining))
        if r_sa2 > best_radius:
            best_radius = r_sa2
            best_c = c_sa2.copy()
    
    # 6. SLSQP polish pass
    if best_c is not None and time.time() < deadline - 8:
        remaining = max(8.0, deadline - time.time() - 8)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(22.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    # 7. Final micro-polish to squeeze out remaining gains
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(3.0, deadline - time.time() - 1)
        c_micro, r_micro = _slsqp_polish(best_c, best_radius, max_time=min(8.0, remaining))
        if r_micro > best_radius:
            best_radius = r_micro
            best_c = c_micro
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
