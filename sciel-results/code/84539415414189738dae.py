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
    """
    Focused energy: only considers pairs within a threshold of the current minimum.
    """
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
    
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            maxiter = 80 if beta < 100 else 150
            
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
    
    return best_c, best_r


def _local_optimize_focused(c0, max_time=20.0, temps=None):
    """
    Local optimization using the focused energy function.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temps is None:
        temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
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
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
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
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    """
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


def _cutting_plane_polish(c0, r0, max_time=25.0):
    """
    Cutting-plane method: approximate the maximum radius function from below
    using active constraints, solve a QP subproblem, and iterate.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    def _get_active_pairs(c, r_target, margin=0.003):
        """Get pairs that are close to the minimum distance."""
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(N):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
    
    def _get_active_boundary(c, r_target, margin=0.003):
        """Get boundary constraints that are close to active."""
        bd = 0.5 - np.abs(c)
        mask = bd < r_target + margin
        return np.where(mask)[0]
    
    # Iterative cutting-plane refinement
    for iteration in range(30):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + 1e-6
        
        pairs = _get_active_pairs(best_c, r_target, margin=0.005)
        boundary_idx = _get_active_boundary(best_c, r_target, margin=0.005)
        
        npairs = len(pairs)
        nbound = len(boundary_idx)
        
        if npairs == 0 and nbound == 0:
            break
        
        # Build the QP subproblem: maximize r subject to active constraints
        # Variables: positions (2N) and r (1)
        # Constraints: for each active pair, |c_i - c_j| >= 2r
        #               for each active boundary, 0.5 - |c_ik| >= r
        
        def _qp_objective(x):
            """Objective: -r (we want to maximize r)."""
            return -x[-1]
        
        def _qp_jacobian(x):
            """Jacobian of objective."""
            grad = np.zeros(2 * N + 1)
            grad[-1] = -1.0
            return grad
        
        # Build inequality constraints: g(x) >= 0
        # For pairs: |c_i - c_j|^2 - 4r^2 >= 0 (non-smooth, use linearized)
        # Linearized: (c_i - c_j)·(c_i - c_j)_0 / |c_i - c_j|_0 - 2r >= 0
        
        def _qp_constraints(x):
            c = x[:2*N].reshape(-1, 2)
            r = x[-1]
            constraints = []
            
            # Pair constraints (linearized)
            if npairs > 0:
                ci = c[pairs[:, 0]]
                cj = c[pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                
                # Linearized: unit_vec · (c_i - c_j) - 2r >= 0
                pair_constr = np.sum(unit_vecs * (ci - cj), axis=1) - 2 * r
                constraints.append(pair_constr)
            
            # Boundary constraints
            if nbound > 0:
                bd = 0.5 - np.abs(c[boundary_idx])
                # For each dimension of active boundary points
                bd_flat = bd.ravel()
                bound_constr = bd_flat - r
                constraints.append(bound_constr)
            
            if constraints:
                return np.concatenate(constraints)
            return np.array([1.0])
        
        def _qp_jac_constraints(x):
            c = x[:2*N].reshape(-1, 2)
            r = x[-1]
            
            total_constr = 0
            if npairs > 0:
                total_constr += npairs
            if nbound > 0:
                total_constr += 2 * nbound
            
            jac = np.zeros((total_constr, 2 * N + 1))
            row = 0
            
            # Pair constraint Jacobians
            if npairs > 0:
                ci = c[pairs[:, 0]]
                cj = c[pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                
                col_i = pairs[:, 0] * 2
                col_j = pairs[:, 1] * 2
                
                jac[row:row+npairs, col_i] = unit_vecs[:, 0]
                jac[row:row+npairs, col_i+1] = unit_vecs[:, 1]
                jac[row:row+npairs, col_j] = -unit_vecs[:, 0]
                jac[row:row+npairs, col_j+1] = -unit_vecs[:, 1]
                jac[row:row+npairs, -1] = -2.0
                
                row += npairs
            
            # Boundary constraint Jacobians
            if nbound > 0:
                for idx in boundary_idx:
                    for dim in range(2):
                        sign = -np.sign(c[idx, dim])
                        if c[idx, dim] == 0:
                            sign = 1.0
                        jac[row, idx*2 + dim] = sign
                        jac[row, -1] = -1.0
                        row += 1
            
            return jac
        
        # Solve the QP subproblem
        x0 = np.concatenate([best_c.flatten(), [r_target]])
        bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N) + [(0.0, 0.5)]
        
        try:
            result = minimize(
                _qp_objective,
                x0,
                jac=_qp_jacobian,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': _qp_constraints, 'jac': _qp_jac_constraints}],
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            
            if result.success or result.fun < 0:
                c_new = np.clip(result.x[:2*N].reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r + 1e-12:
                    best_c = c_new
                    best_r = r_new
                elif r_new < best_r - 1e-8:
                    # Move got worse, try smaller step
                    break
        except Exception:
            break
        
        # Check if we've converged
        if best_r < r_target - 1e-8:
            break
    
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    def _get_pairs_vectorized(c, r_target, margin=0.005):
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
    
    step = 5e-5
    max_iter = 80
    
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + step
        
        pairs = _get_pairs_vectorized(best_c, r_target, margin=0.008)
        npairs = len(pairs)
        
        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            
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
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            
            signs = -np.sign(c)
            for i in range(n):
                jac[2*i, 2*i] = signs[i, 0]
                jac[2*i+1, 2*i+1] = signs[i, 1]
            
            if npairs == 0:
                return jac
            
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            
            row_indices = 2 * n + np.arange(npairs)
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
                options={'maxiter': 150, 'ftol': 1e-14, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.05, 2e-4)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
                    
        except Exception:
            break
            
    return best_c, best_r


def _ultra_polish(c0, r0, max_time=10.0):
    """
    Ultra-precise final refinement using very low temperature focused energy.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    temps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
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
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.49999, 0.49999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with smooth objective (full soft-min)
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(30.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Focused energy optimization (faster per iteration, more temperature levels)
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(8.0, deadline - time.time() - 40)
        c_foc, r_foc = _local_optimize_focused(best_c, max_time=min(18.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()
    
    # 4. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 30:
        remaining = max(10.0, deadline - time.time() - 30)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(20.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 5. Another smooth optimization pass from the SA result
    if best_c is not None and time.time() < deadline - 22:
        remaining = max(5.0, deadline - time.time() - 22)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(12.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # 6. Focused energy refinement after SA
    if best_c is not None and time.time() < deadline - 15:
        remaining = max(5.0, deadline - time.time() - 15)
        c_foc2, r_foc2 = _local_optimize_focused(best_c, max_time=min(8.0, remaining))
        if r_foc2 > best_radius:
            best_radius = r_foc2
            best_c = c_foc2.copy()
    
    # 7. NEW: Cutting-plane polish - approximates max radius from below
    if best_c is not None and time.time() < deadline - 8:
        remaining = max(5.0, deadline - time.time() - 8)
        c_cp, r_cp = _cutting_plane_polish(best_c, best_radius, max_time=min(10.0, remaining))
        if r_cp > best_radius:
            best_radius = r_cp
            best_c = c_cp.copy()
    
    # 8. Final high-precision SLSQP polish
    if best_c is not None and time.time() < deadline - 4:
        remaining = max(4.0, deadline - time.time() - 4)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(8.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    # 9. Ultra-precise final refinement
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(1.5, deadline - time.time() - 0.5)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(4.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
