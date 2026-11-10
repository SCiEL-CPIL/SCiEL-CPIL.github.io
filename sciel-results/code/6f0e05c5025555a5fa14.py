"""Rewritten program for packing 50 circles in a unit circle."""
import numpy as np
import time

N = 50

def _hexagonal_lattice(n_points):
    """Generate a hexagonal lattice of points centered at origin."""
    points = []
    max_r = int(np.sqrt(n_points / 0.75)) + 3
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    if len(radii) > 0:
        max_rad = np.max(radii)
        points = points * (0.95 / max_rad)
        
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii, kind='stable')
    return points[indices[:n_points]]

def _compute_softmin_gradient(c, beta):
    N = len(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    
    d_pairs = dist[mask] / 2.0
    d_bound = 1.0 - np.linalg.norm(c, axis=1)
    
    all_x = np.concatenate([d_pairs, d_bound])
    x_min = np.min(all_x)
    shifted = all_x - x_min
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    value = -x_min - (1.0 / beta) * np.log(sum_exp)
    weights = exp_terms / sum_exp
    
    w_pairs = weights[:len(d_pairs)]
    w_bound = weights[len(d_pairs):]
    
    grad = np.zeros_like(c)
    W_pair = np.zeros((N, N))
    W_pair[mask] = w_pairs
    
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    force_vec = diff / (2.0 * safe_dist[:, :, np.newaxis])
    force_vec[np.eye(N, dtype=bool)] = 0.0
    
    grad_pair = -(W_pair * force_vec).sum(axis=1)
    
    c_norm = np.linalg.norm(c, axis=1, keepdims=True)
    safe_c_norm = np.where(c_norm > 1e-10, c_norm, 1.0)
    grad_bound = w_bound[:, np.newaxis] * c / safe_c_norm
    
    grad = grad_pair + grad_bound
    return value, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_c = None
    best_r = 0.0
    
    c = _hexagonal_lattice(N)
    
    beta = 20.0
    step_size = 0.05
    
    mask = ~np.eye(N, dtype=bool)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart == 0:
            c = _hexagonal_lattice(N)
        elif best_c is not None and restart % 3 == 0:
            c = best_c + np.random.normal(0, 0.02, (N, 2))
        else:
            c = _hexagonal_lattice(N) + np.random.normal(0, 0.1, (N, 2))
            
        norms = np.linalg.norm(c, axis=1)
        c = c * np.minimum(1.0, 0.95 / np.maximum(norms, 1e-6))[:, np.newaxis]
        
        current_r = 0.0
        no_improve_count = 0
        max_no_improve = 30
        beta = 20.0
        step_size = 0.05
        
        for it in range(3000):
            if time.time() > deadline:
                break
                
            r, grad = _compute_softmin_gradient(c, beta)
            c_new = c + step_size * grad
            
            r_new, _ = _compute_softmin_gradient(c_new, beta)
            
            if r_new > r:
                c = c_new
                step_size *= 1.1
                if r_new > current_r:
                    current_r = r_new
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                step_size *= 0.7
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                beta *= 1.2
                step_size = 0.05
                no_improve_count = 0
                if beta > 200:
                    break
        
        dists = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        min_pair = np.min(dists[mask]) / 2.0
        min_bound = 1.0 - np.max(np.linalg.norm(c, axis=1))
        r_hard = min(min_pair, min_bound)
        
        if r_hard > best_r:
            best_r = r_hard
            best_c = c.copy()

    if best_c is None:
        return _hexagonal_lattice(N)
        
    return best_c
