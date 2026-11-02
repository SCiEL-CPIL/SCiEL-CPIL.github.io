"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using sparse K-NN Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _get_knn_pairs(c, K=8):
    """Get K nearest neighbor pairs using cKDTree."""
    tree = cKDTree(c)
    # Query K+1 neighbors (including self), take indices 1:K+1
    dists, indices = tree.query(c, k=K+1)
    # dists: (N, K+1), indices: (N, K+1)
    # Take columns 1:K+1 (skip self)
    d_knn = dists[:, 1:]  # (N, K)
    idx_knn = indices[:, 1:]  # (N, K)
    
    # Collect unique pairs
    pairs = set()
    for i in range(N):
        for k in range(K):
            j = idx_knn[i, k]
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    
    pair_list = sorted(pairs)
    if len(pair_list) == 0:
        return np.array([], dtype=int).reshape(0, 2)
    return np.array(pair_list, dtype=int)


def _sparse_energy(params, beta=200.0, K=8, pair_cache=None):
    """
    Sparse K-NN based log-sum-exp energy.
    pair_cache: optional (pairs, last_update_iter) to avoid recomputing KNN every call.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Get or use cached pairs
    if pair_cache is not None and pair_cache[1] is not None:
        pairs = pair_cache[0]
    else:
        pairs = _get_knn_pairs(c, K=K)
    
    n_pairs = len(pairs)
    
    if n_pairs == 0:
        # Fallback: no pairs, just wall energy
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        eps = 1e-8
        d_left_safe = np.clip(d_left, eps, None)
        d_right_safe = np.clip(d_right, eps, None)
        d_down_safe = np.clip(d_down, eps, None)
        d_up_safe = np.clip(d_up, eps, None)
        E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
        grad_x = -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
        grad_y = -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
        return E_wall, np.concatenate([grad_x, grad_y])
    
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    d_pairs = np.sqrt(dx**2 + dy**2)
    
    # Log-Sum-Exp
    d_min = np.min(d_pairs)
    d_shifted = d_pairs - d_min
    weights = np.exp(-beta * d_shifted)
    w_sum = np.sum(weights)
    E_pairs = np.log(w_sum) / beta
    
    p = weights / w_sum
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    np.add.at(grad_x, i_idx, -p * dx * inv_d)
    np.add.at(grad_x, j_idx, p * dx * inv_d)
    np.add.at(grad_y, i_idx, -p * dy * inv_d)
    np.add.at(grad_y, j_idx, p * dy * inv_d)
    
    # Wall energy
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    E_total = E_wall + E_pairs
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0
    
    # Try different densities with finer granularity around optimal region
    for d in np.linspace(0.052, 0.075, 40):
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
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, time_budget=20.0, beta=100.0, K=8):
    """Run L-BFGS-B with sparse K-NN Log-Sum-Exp energy."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    # Precompute initial KNN pairs
    pairs = _get_knn_pairs(c0, K=K)
    pair_cache = [pairs, 0]
    iter_count = [0]
    
    try:
        def fun_and_grad(params):
            iter_count[0] += 1
            # Update KNN pairs every 30 iterations
            if iter_count[0] % 30 == 0:
                c_cur = params.reshape(N, 2)
                pair_cache[0] = _get_knn_pairs(c_cur, K=K)
            return _sparse_energy(params, beta=beta, K=K, pair_cache=pair_cache)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 10000, 'maxls': 100}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

def _slsqp_active_polish(c, r_target, time_limit=5.0):
    """
    Uses SLSQP to find a configuration where all active constraints are satisfied
    for a target radius r_target.
    """
    start_time = time.time()
    c = c.copy()
    
    # Identify active constraints
    tol = 1e-8
    dists = pdist(c, metric='euclidean')
    dist_mat = squareform(dists)
    
    # Active pairs: distance < 2*r_target + tol
    active_mask = dist_mat < (2 * r_target + tol)
    np.fill_diagonal(active_mask, False)
    
    active_pairs = np.argwhere(active_mask)
    active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
    
    # Active bounds
    bdists = np.minimum(
        np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
        np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
    )
    active_bounds = np.where(bdists < (r_target + tol))[0]
    
    if len(active_pairs) == 0 and len(active_bounds) == 0:
        return c, False

    cons = []
    for i, j in active_pairs:
        def constraint_pair(x, i=i, j=j):
            xi = x[2*i:2*i+2]
            xj = x[2*j:2*j+2]
            return np.dot(xi-xj, xi-xj) - (2*r_target)**2
        cons.append({'type': 'ineq', 'fun': constraint_pair})
        
    for i in active_bounds:
        d_left = 0.5 - c[i, 0]
        d_right = c[i, 0] + 0.5
        d_down = 0.5 - c[i, 1]
        d_up = c[i, 1] + 0.5
        min_d = min(d_left, d_right, d_down, d_up)
        
        if d_left == min_d:
            def constraint_wall_left(x, i=i):
                return 0.5 - x[2*i] - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_left})
        elif d_right == min_d:
            def constraint_wall_right(x, i=i):
                return x[2*i] + 0.5 - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_right})
        elif d_down == min_d:
            def constraint_wall_down(x, i=i):
                return 0.5 - x[2*i+1] - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_down})
        else:
            def constraint_wall_up(x, i=i):
                return x[2*i+1] + 0.5 - r_target
            cons.append({'type': 'ineq', 'fun': constraint_wall_up})

    bounds = [(-0.5 + r_target, 0.5 - r_target)] * (2 * N)
    
    def objective(x):
        return 0.0

    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 50, 'ftol': 1e-12}
        )
        
        if res.success:
            c_new = res.x.reshape(N, 2)
            r_new = _compute_radius(c_new)
            if r_new >= r_target - 1e-10:
                return c_new, True
    except Exception:
        pass
        
    return c, False

def _local_polish(c, time_limit=30.0):
    """
    Fast local polish using active-constraint gradient ascent on radius.
    Uses a smooth penalty-based approach for speed.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    def _rigid_body_perturb(c, rng, radius=0.15, max_clusters=5):
        """
        Move clusters of nearby circles simultaneously to preserve local structure.
        Returns a perturbed copy of c.
        """
        c_new = c.copy()
        n = len(c_new)
        visited = np.zeros(n, dtype=bool)
        
        # Find clusters based on proximity
        dx = c_new[:, np.newaxis, 0] - c_new[np.newaxis, :, 0]
        dy = c_new[:, np.newaxis, 1] - c_new[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            # BFS to find connected component within radius
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                j = queue.pop()
                neighbors = np.where(dist[j] < radius)[0]
                for k in neighbors:
                    if not visited[k]:
                        visited[k] = True
                        cluster.append(k)
                        queue.append(k)
            clusters.append(cluster)
        
        # Select up to max_clusters clusters to perturb
        if len(clusters) > max_clusters:
            idx = rng.choice(len(clusters), size=max_clusters, replace=False)
            selected_clusters = [clusters[i] for i in idx]
        else:
            selected_clusters = clusters
        
        for cluster in selected_clusters:
            if len(cluster) < 2:
                continue
            # Generate a random direction and magnitude
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = rng.uniform(0.003, 0.015)
            dx_move = magnitude * np.cos(angle)
            dy_move = magnitude * np.sin(angle)
            
            for i in cluster:
                c_new[i, 0] += dx_move
                c_new[i, 1] += dy_move
        
        # Clip to stay inside the square
        c_new = np.clip(c_new, -0.49999, 0.49999)
        return c_new
    
    def _annealed_polish(c, start_time, time_limit, beta_start=150.0, beta_end=1200.0):
        """
        Multi-temperature annealing polish with rigid-body perturbations.
        """
        c = c.copy()
        current_r = _compute_radius(c)
        rng = np.random.RandomState(42)
        
        # Temperature schedule: more levels for gradual refinement
        n_steps = 8
        betas = np.geomspace(beta_start, beta_end, n_steps)
        
        for beta in betas:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Run L-BFGS-B at this temperature
            try:
                res = minimize(
                    lambda p: _polish_energy(p, beta=beta),
                    c.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.49999, 0.49999)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 600}
                )
                c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
                r_opt = _compute_radius(c_opt)
                if r_opt > current_r:
                    c = c_opt
                    current_r = r_opt
            except Exception:
                pass
            
            # Rigid-body perturbation to escape local minima
            if time.time() - start_time > time_limit - 10:
                break
            
            c_perturbed = _rigid_body_perturb(c, rng, radius=0.10, max_clusters=4)
            r_perturbed = _compute_radius(c_perturbed)
            
            # Accept if better or with some probability (simulated annealing style)
            if r_perturbed > current_r:
                c = c_perturbed
                current_r = r_perturbed
            else:
                # Accept worse solutions with decreasing probability
                delta_r = current_r - r_perturbed
                T = 0.00008 * (1.0 - (time.time() - start_time) / time_limit)
                if delta_r < T and rng.random() < np.exp(-delta_r / max(T, 1e-10)):
                    c = c_perturbed
                    current_r = r_perturbed
        
        return c, current_r
    
    def _polish_energy(params, beta=300.0):
        """Sparse K-NN energy for polish phase."""
        c_ = params.reshape(N, 2)
        x = c_[:, 0]
        y = c_[:, 1]
        
        # Use sparse K-NN
        pairs = _get_knn_pairs(c_, K=10)
        n_pairs = len(pairs)
        
        if n_pairs == 0:
            d_left = 0.5 - x
            d_right = 0.5 + x
            d_down = 0.5 - y
            d_up = 0.5 + y
            eps = 1e-8
            d_left_safe = np.clip(d_left, eps, None)
            d_right_safe = np.clip(d_right, eps, None)
            d_down_safe = np.clip(d_down, eps, None)
            d_up_safe = np.clip(d_up, eps, None)
            E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
            grad_x = -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
            grad_y = -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
            return E_wall, np.concatenate([grad_x, grad_y])
        
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d_pairs = np.sqrt(dx**2 + dy**2)
        
        d_min = np.min(d_pairs)
        d_shifted = d_pairs - d_min
        w_pairs = np.exp(-beta * d_shifted)
        w_sum = np.sum(w_pairs)
        E_pairs = np.log(w_sum) / beta
        p_pairs = w_pairs / w_sum
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        np.add.at(grad_x, i_idx, -p_pairs * dx * inv_d)
        np.add.at(grad_x, j_idx, p_pairs * dx * inv_d)
        np.add.at(grad_y, i_idx, -p_pairs * dy * inv_d)
        np.add.at(grad_y, j_idx, p_pairs * dy * inv_d)
        
        # Boundary: use soft-min
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        
        d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
        d_min_b = np.min(d_all, axis=1, keepdims=True)
        wall_weights = np.exp(-beta * (d_all - d_min_b))
        wall_weights /= wall_weights.sum(axis=1, keepdims=True)
        
        E_bounds = np.log(np.sum(wall_weights)) / beta
        
        p_bounds = wall_weights[:, 0] + wall_weights[:, 1] + wall_weights[:, 2] + wall_weights[:, 3]
        p_bounds_sum = np.sum(p_bounds)
        if p_bounds_sum > 0:
            p_bounds_norm = p_bounds / p_bounds_sum
            grad_x += p_bounds_norm * (-wall_weights[:, 0] + wall_weights[:, 1])
            grad_y += p_bounds_norm * (-wall_weights[:, 2] + wall_weights[:, 3])
        
        E = E_pairs + E_bounds
        total_grad = np.concatenate([grad_x, grad_y])
        return E, total_grad
    
    # Run annealed polish with rigid-body perturbations
    if time.time() - start_time < time_limit * 0.75:
        c_annealed, r_annealed = _annealed_polish(c, start_time, time_limit * 0.75, beta_start=150.0, beta_end=1200.0)
        if r_annealed > current_r:
            c = c_annealed
            current_r = r_annealed
    
    # Final high-precision polish with very high beta
    if time.time() - start_time < time_limit - 5:
        try:
            res = minimize(
                lambda p: _polish_energy(p, beta=2000.0),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 8000}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
    
    # SLSQP Active-Set Polish
    if time.time() - start_time < time_limit - 1:
        current_r = _compute_radius(c)
        # Try to increase radius in small steps using SLSQP
        for step in [1e-6, 5e-7, 2e-7, 1e-7, 5e-8]:
            if time.time() - start_time > time_limit - 1:
                break
            target_r = current_r + step
            c_new, success = _slsqp_active_polish(c, target_r, time_limit=5.0)
            if success:
                c = c_new
                current_r = _compute_radius(c)
            else:
                # If it fails, we might be at the limit or SLSQP failed.
                # Try smaller step?
                # For now, break to save time.
                break
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 148
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Run global optimization with multiple beta values
    for beta in [40.0, 80.0, 120.0, 180.0]:
        if time.time() > deadline - 80:
            break
        c_opt = _optimize_energy(best_c, time_budget=18.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Try multiple perturbed restarts if time allows
    remaining = deadline - time.time() - 60
    if remaining > 20:
        rng = np.random.RandomState(42)
        for attempt in range(4):
            if time.time() > deadline - 60:
                break
            perturbation = rng.randn(N, 2) * 0.008
            c_perturbed = np.clip(best_c + perturbation, -0.4999, 0.4999)
            
            c_opt2 = _optimize_energy(c_perturbed, time_budget=12.0, beta=100.0)
            r_opt2 = _compute_radius(c_opt2)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()
    
    # Local polish - use most of the remaining time
    remaining = deadline - time.time() - 5
    if remaining > 15:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Second polish pass if time allows
    remaining = deadline - time.time() - 3
    if remaining > 15:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Third polish pass with even tighter tolerances
    remaining = deadline - time.time() - 2
    if remaining > 10:
        c_polished3, r_polished3 = _local_polish(best_c, time_limit=min(remaining, 25.0))
        if r_polished3 > best_r:
            best_r = r_polished3
            best_c = c_polished3.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
