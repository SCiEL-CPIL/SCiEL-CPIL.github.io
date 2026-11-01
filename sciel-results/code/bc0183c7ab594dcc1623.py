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
    
    return points

def _radial_ring_init(n_points):
    """Initialize points on concentric rings."""
    points = []
    # Approximate number of rings
    # Area ~ pi, ring area ~ 2*pi*r*dr. 
    # Let's just place points in rings with increasing radius.
    # Simple heuristic: distribute points in rings.
    
    # A better way: use a spiral or specific ring counts.
    # For N=300, let's try rings of 1, 6, 12, 18... (hexagonal close packing approximation)
    # Or just simple uniform rings.
    
    # Let's use a simple approach: 
    # r_k = sqrt(k/N) * R_container
    # But we need to distribute angles.
    
    # Actually, a "sunflower" or Vogel model is good for initial spread.
    # But for packing, rings are better.
    
    # Let's try to fit hexagonal rings.
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # ...
    # Ring k: 6k points
    
    remaining = n_points
    ring_idx = 0
    current_count = 1
    points.append([0.0, 0.0])
    remaining -= 1
    
    while remaining > 0:
        ring_idx += 1
        count_in_ring = 6 * ring_idx
        
        # Adjust count to not exceed remaining
        if count_in_ring > remaining:
            count_in_ring = remaining
            
        # Radius for this ring
        # Approximate radius based on area or just linear scaling
        # r ~ sqrt(ring_idx)
        r = np.sqrt(ring_idx / (n_points / 6.0)) * 0.9 # Scale factor to stay inside
        
        # Ensure it's inside
        if r > 0.95:
             r = 0.95
             
        for i in range(count_in_ring):
            angle = 2 * np.pi * i / count_in_ring
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
            
        remaining -= count_in_ring
        
    points = np.array(points)
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
    x is a flat array of shape (2*N,).
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs = L_safe[i_idx, j_idx] / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Gradient
    grad = np.zeros_like(C)
    
    # Pairwise terms
    P_pairs = P[:len(i_idx)]
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    U = D_flat / L_flat_safe[:, np.newaxis]
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall terms
    P_walls = P[len(i_idx):]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_soft_min(points_init, time_limit):
    """
    Optimize the packing using soft-min with annealing beta.
    """
    start_time = time.time()
    best_points = points_init.copy()
    best_score = _compute_true_score(points_init)
    current_points = points_init.copy()
    
    # Beta schedule
    betas = [10, 50, 200, 1000, 5000, 20000, 100000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Define objective
        def obj_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            # Use L-BFGS-B
            # Increase maxiter as beta increases to allow convergence
            max_iter = 200
            res = minimize(obj_grad, current_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
            
            new_points = res.x.reshape(N, 2)
            
            # Check validity
            if not np.all(np.isfinite(new_points)):
                continue
                
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
            # Always update current_points if valid, even if score didn't improve slightly,
            # because we are moving towards the minimum of the soft objective which approximates the hard one.
            if np.all(np.isfinite(new_points)):
                current_points = new_points.copy()
                
        except Exception:
            # If optimization fails, keep current points
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for 150s total
    
    # Multi-start strategy
    # 1. Hex lattice
    # 2. Rotated Hex lattice
    # 3. Radial rings
    
    initializers = [
        lambda: _hex_lattice_init(N, rotation=0.0),
        lambda: _hex_lattice_init(N, rotation=np.pi/12), # 15 degrees
        lambda: _radial_ring_init(N)
    ]
    
    best_points = None
    best_score = 0.0
    
    for i, init_func in enumerate(initializers):
        elapsed = time.time() - start_time
        # Allocate remaining time to each initializer, leaving a buffer
        # If we have 3 initializers, give each ~1/3 of the time, but ensure the last one has enough to polish.
        # Better: Run first two with less time, last one with more?
        # Or just run them sequentially and check time.
        
        # Let's say we spend 40s on first, 40s on second, 60s on third (polish).
        # Or simpler: run each with a time slice.
        
        if i == 0:
            slice_time = 40.0
        elif i == 1:
            slice_time = 40.0
        else:
            slice_time = time_limit - (time.time() - start_time) - 2.0
        
        if slice_time < 5.0:
            break
            
        try:
            points_init = init_func()
            # Optimize
            points_opt, score_opt = _optimize_soft_min(points_init, slice_time)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
        except Exception:
            continue
            
    # If no valid points were found (should not happen), fallback to hex
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_true_score(best_points)
        
    # Final polish with very high beta and tight tolerance
    elapsed = time.time() - start_time
    if elapsed < time_limit - 2.0:
        # Use a single high-beta polish on the best result
        # We can reuse _optimize_soft_min with a high beta start, or just a direct L-BFGS-B call
        # Let's do a direct call for precision
        beta_final = 50000
        def obj_grad_final(x):
            return _soft_min_obj_and_grad(x, beta_final)
            
        try:
            res = minimize(obj_grad_final, best_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            new_points = res.x.reshape(N, 2)
            new_score = _compute_true_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init(N)
        
    # Verify all points are inside the unit circle
    dists = np.linalg.norm(best_points, axis=1)
    if np.max(dists) > 1.0:
        scale = 0.99 / np.max(dists)
        best_points = best_points * scale
        
    return best_points
