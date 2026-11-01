"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of distances (not squared).
    This is a smooth approximation of -min(d_i).
    We also add a wall penalty to keep points inside.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances (not squared for better approximation of min)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    # Upper triangle indices for pairs
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Log-Sum-Exp trick for stability
    d_shifted = d_pairs - np.min(d_pairs)
    weights = np.exp(-beta * d_shifted)
    p = weights / np.sum(weights)
    
    # Energy: (1/beta) * log(sum(exp(-beta * d_i)))
    # This is LARGE when distances are small, so minimizing spreads points.
    E_pairs = np.log(np.sum(weights)) / beta
    
    # Gradient computation
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    # dE/d(d_k) = -1 (from log-sum-exp form)
    # d(d_k)/d(c_i_x) = (c_i_x - c_j_x) / d_k
    # So dE/d(c_i_x) = -sum_k p_k * (c_i_x - c_j_x) / d_k
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    grad_x[i_idx] -= p * vec_x * inv_d
    grad_x[j_idx] += p * vec_x * inv_d
    
    grad_y[i_idx] -= p * vec_y * inv_d
    grad_y[j_idx] += p * vec_y * inv_d
    
    # Wall Penalty - use soft-min for boundaries too
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    # Soft min of boundary distances
    d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)  # (N, 4)
    d_min = np.min(d_all, axis=1, keepdims=True)
    wall_weights = np.exp(-beta * (d_all - d_min))  # (N, 4)
    wall_weights /= wall_weights.sum(axis=1, keepdims=True)
    
    # E_bounds = (1/beta) * log(sum(exp(-beta * min_wall_dist)))
    # But we already have the soft min via wall_weights
    E_bounds = -np.log(np.mean(wall_weights[:, 0] + wall_weights[:, 1] + wall_weights[:, 2] + wall_weights[:, 3])) / beta
    
    # Actually, let's use a simpler boundary energy
    # Penalize points that are too close to any wall
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    # Use inverse square penalty for strong repulsion from walls
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    # Gradient of wall part
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    # Combine energies
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

def _optimize_energy(c0, time_budget=20.0, beta=100.0):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        # Wrap to handle beta
        def fun_and_grad(params):
            return _soft_min_energy(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 50}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

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
        """Energy that maximizes the minimum distance using log-sum-exp."""
        c_ = params.reshape(N, 2)
        x = c_[:, 0]
        y = c_[:, 1]
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        # Boundary distances
        bdists = np.minimum(
            np.minimum(0.5 - x, 0.5 + x),
            np.minimum(0.5 - y, 0.5 + y)
        )
        
        # We want to maximize min(d_pairs/2, bdists)
        # Use log-sum-exp to get a smooth approximation
        # E = (1/beta) * log(sum(exp(-beta * d)))
        
        # For pairs (use d/2 since radius = d/2)
        d_shifted = d_pairs - np.min(d_pairs)
        w_pairs = np.exp(-beta * d_shifted)
        E_pairs = np.log(np.sum(w_pairs)) / beta
        
        # For boundaries
        bd_shifted = bdists - np.min(bdists)
        w_bounds = np.exp(-beta * bd_shifted)
        E_bounds = np.log(np.sum(w_bounds)) / beta
        
        E = E_pairs + E_bounds
        
        # Gradient
        p_pairs = w_pairs / np.sum(w_pairs)
        p_bounds = w_bounds / np.sum(w_bounds)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        # dE/d(d_k) = -1
        # d(d_k)/d(c_i) = (c_i - c_j) / d_k
        grad_x[i_idx] -= p_pairs * vec_x * inv_d
        grad_x[j_idx] += p_pairs * vec_x * inv_d
        grad_y[i_idx] -= p_pairs * vec_y * inv_d
        grad_y[j_idx] += p_pairs * vec_y * inv_d
        
        # Boundary gradient
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        
        d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
        d_min = np.min(d_all, axis=1, keepdims=True)
        wall_weights = np.exp(-beta * (d_all - d_min))
        wall_weights /= wall_weights.sum(axis=1, keepdims=True)
        
        grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
        grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
        
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
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 1500}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
    
    # Final micro-adjustment: try to nudge active points
    if time.time() - start_time < time_limit - 1:
        for _ in range(100):
            if time.time() - start_time > time_limit - 1:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            
            min_pair_dist = np.min(dist_matrix)
            min_boundary_dist = np.min(bdists)
            target_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if target_r > current_r + 1e-15:
                current_r = target_r
            
            # Find active constraints with tighter tolerance
            tol = 1e-12
            active_pairs = np.argwhere(dist_matrix < min_pair_dist + tol)
            active_bounds = np.where(bdists < min_boundary_dist + tol)[0]
            
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                break
            
            # Build force vectors
            forces = np.zeros((N, 2))
            
            for i, j in active_pairs:
                if i == j:
                    continue
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-14:
                    unit_vec = vec / norm
                    forces[i] += unit_vec
                    forces[j] -= unit_vec
            
            for i in active_bounds:
                d_left = 0.5 - c[i, 0]
                d_right = c[i, 0] + 0.5
                d_down = 0.5 - c[i, 1]
                d_up = c[i, 1] + 0.5
                
                min_d = min(d_left, d_right, d_down, d_up)
                if d_left == min_d:
                    forces[i, 0] += 1.0
                elif d_right == min_d:
                    forces[i, 0] -= 1.0
                elif d_down == min_d:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
            
            # Try moving each active point with smaller steps
            moved = False
            for step in [5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8]:
                if moved:
                    break
                for i in range(N):
                    if np.linalg.norm(forces[i]) > 1e-14:
                        direction = forces[i] / np.linalg.norm(forces[i])
                        new_pos = c[i] + direction * step
                        if np.all(np.abs(new_pos) < 0.499999):
                            old_pos = c[i].copy()
                            c[i] = new_pos
                            new_r = _compute_radius(c)
                            if new_r > current_r:
                                current_r = new_r
                                moved = True
                            else:
                                c[i] = old_pos
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Run global optimization with multiple beta values
    for beta in [40.0, 80.0, 120.0, 180.0]:
        if time.time() > deadline - 90:
            break
        c_opt = _optimize_energy(best_c, time_budget=18.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Try multiple perturbed restarts if time allows
    remaining = deadline - time.time() - 70
    if remaining > 25:
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if time.time() > deadline - 70:
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
