import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    points += np.random.normal(0, s * 0.01, size=points.shape)
    
    return points

def _radial_ring_init(n_points):
    """Generate points on concentric rings."""
    points = []
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    n_rings = int(1.0 / s) + 1
    
    for r_idx in range(n_rings):
        r = (r_idx + 0.5) * s
        if r >= 1.0:
            break
        n_on_ring = max(1, int(2 * np.pi * r / s))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        while len(points) < n_points:
            r = np.sqrt(np.random.random())
            a = np.random.random() * 2 * np.pi
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    elif len(points) > n_points:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n_points]]
    
    points += np.random.normal(0, s * 0.01, size=points.shape)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    Uses O(N^2) computation but avoids storing the full distance matrix twice.
    """
    C = x.reshape(N, 2)
    
    # Compute pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.sqrt(np.sum(D * D, axis=2))
    
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = L_safe[i_idx, j_idx] / 2.0
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-12})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _gd_polish(points, lr=0.001, n_iter=100, beta=500.0):
    """Gradient descent polish pass with adaptive learning rate."""
    current = points.copy()
    best_val = np.inf
    best_current = current.copy()
    for i in range(n_iter):
        val, grad = _soft_min_obj_and_grad(current.flatten(), beta)
        if val < best_val:
            best_val = val
            best_current = current.copy()
        
        # Adaptive step: reduce if objective worsens
        step = lr * (0.95 ** (i // 20))
        new_current = current - step * grad.reshape(N, 2)
        
        # Check if step improved the objective
        new_val, _ = _soft_min_obj_and_grad(new_current.flatten(), beta)
        if new_val > val:
            # Backtrack
            step *= 0.5
            new_current = current - step * grad.reshape(N, 2)
        
        current = new_current
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            current[mask] = current[mask] * (0.999 / norms[mask])[:, np.newaxis]
    
    return best_current

def _anneal_beta(points, beta_start, beta_end, n_steps, time_limit, max_iter_per_step=100):
    """Progressively increase beta and optimize at each step."""
    current = points.copy()
    for i in range(n_steps):
        beta = beta_start * (beta_end / beta_start) ** (i / max(n_steps - 1, 1))
        current = _optimize_lbfgs(current, beta, max_iter=max_iter_per_step, time_limit=time_limit)
    return current

def _local_refine(points, n_iter=200, time_budget=10.0):
    """Local refinement: repeatedly move each point to maximize its local clearance."""
    start_time = time.time()
    current = points.copy()
    
    for iteration in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                    break
            
            # Find nearest neighbor distance and wall distance
            dists_to_others = np.linalg.norm(current[i] - current, axis=1)
            dists_to_others[i] = np.inf
            min_pair_dist = np.min(dists_to_others)
            
            wall_dist = 1.0 - np.linalg.norm(current[i])
            
            current_radius = min(min_pair_dist / 2.0, wall_dist)
            
            # Try small perturbations in random directions
            best_pos = current[i].copy()
            best_radius = current_radius
            
            for _ in range(20):
                direction = np.random.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-15)
                
                for step_size in [0.001, 0.005, 0.01, 0.02]:
                    candidate = current[i] + direction * step_size
                    if np.linalg.norm(candidate) >= 1.0:
                        continue
                    
                    dists_to_others = np.linalg.norm(candidate - current, axis=1)
                    dists_to_others[i] = np.inf
                    min_pair_dist = np.min(dists_to_others)
                    wall_dist = 1.0 - np.linalg.norm(candidate)
                    candidate_radius = min(min_pair_dist / 2.0, wall_dist)
                    
                    if candidate_radius > best_radius:
                        best_radius = candidate_radius
                        best_pos = candidate
            
            if np.linalg.norm(best_pos - current[i]) > 1e-10:
                current[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin before 150s kill
    
    inits = []
    
    for rot in [0.0, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 2]:
        inits.append(_hex_lattice_init(N, rotation=rot))
    
    inits.append(_radial_ring_init(N))
    
    r = np.sqrt(np.random.random(N))
    a = np.random.random(N) * 2 * np.pi
    inits.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    
    best_points = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() - start_time > time_limit - 30:
            break
        
        current = init.copy()
        
        # Phase 1: Low beta to establish rough structure
        remaining = time_limit - (time.time() - start_time) - 25
        if remaining < 5:
            continue
        current = _optimize_lbfgs(current, 20.0, max_iter=100, time_limit=min(8.0, remaining * 0.2))
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # Phase 2: Medium beta
        remaining = time_limit - (time.time() - start_time) - 20
        if remaining < 5:
            continue
        current = _optimize_lbfgs(current, 100.0, max_iter=150, time_limit=min(10.0, remaining * 0.3))
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # Phase 3: High beta
        remaining = time_limit - (time.time() - start_time) - 15
        if remaining < 5:
            continue
        current = _optimize_lbfgs(current, 500.0, max_iter=200, time_limit=min(12.0, remaining * 0.4))
        remaining = time_limit - (time.time() - start_time) - 10
        if remaining > 3:
            current = _gd_polish(current, lr=0.0005, n_iter=min(50, int(remaining * 5)), beta=1000.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Second pass: refine the best solution further
    if best_points is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 20:
            current = best_points.copy()
            for beta in [1000.0, 2000.0, 5000.0]:
                if time.time() - start_time > time_limit - 15:
                    break
                step_time = min(12.0, (time_limit - 15 - (time.time() - start_time)) / 2)
                if step_time < 2:
                    break
                current = _optimize_lbfgs(current, beta, max_iter=300, time_limit=step_time)
                current = _gd_polish(current, lr=0.0002, n_iter=min(80, int((time_limit - (time.time() - start_time)) * 3)), beta=beta * 2)
                score = _compute_true_score(current)
                if score > best_score:
                    best_score = score
                    best_points = current.copy()
        
        # Local refinement
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10:
            current = best_points.copy()
            current = _local_refine(current, n_iter=min(100, int(remaining * 3)), time_budget=min(8.0, remaining - 5))
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Final high-precision polish
    if best_points is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 8:
            current = best_points.copy()
            current = _optimize_lbfgs(current, 10000.0, max_iter=500, time_limit=min(15.0, remaining - 5))
            remaining = time_limit - (time.time() - start_time)
            if remaining > 3:
                current = _gd_polish(current, lr=0.0001, n_iter=min(150, int(remaining * 5)), beta=20000.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Ensure all points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, np.newaxis]
    
    return best_points
