import numpy as np
import time

N = 300

def _hex_init(n):
    """Generate hexagonal lattice points inside unit disk."""
    s = 0.08 # Smaller spacing to ensure enough points
    points = []
    for j in range(-25, 25):
        for i in range(-25, 25):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y < 0.98: # Keep slightly inside
                points.append([x, y])
    points = np.array(points)
    
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    else:
        # Pad with random points if needed (unlikely)
        while len(points) < n:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1:
                points = np.vstack([points, p])
    return points

def _compute_hard_score(points):
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    radii = np.linalg.norm(points, axis=1)
    min_bound = np.min(1.0 - radii)
    
    return min(min_pair, min_bound)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    np.random.seed(123)
    points = _hex_init(N)
    
    # Optimization
    beta = 5.0
    alpha = 0.02
    noise_scale = 0.005
    
    best_score = _compute_hard_score(points)
    best_points = points.copy()
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        # Compute Gradient
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists[dist == 0] = 1e-8
        
        # Pairwise
        E_pair = np.exp(-beta * dists / 2.0)
        np.fill_diagonal(E_pair, 0.0)
        U = diff / dists[..., np.newaxis]
        grad_pair = np.sum(E_pair * U, axis=1)
        
        # Boundary
        radii = np.linalg.norm(points, axis=1)
        radii_safe = np.where(radii > 1e-8, radii, 1e-8)
        E_bound = np.exp(-beta * (1.0 - radii))
        grad_bound = E_bound[:, np.newaxis] * (-points / radii_safe[:, np.newaxis])
        
        S = np.sum(E_pair) + np.sum(E_bound)
        if S < 1e-10:
            break
            
        grad = (grad_pair + grad_bound) / S
        
        # Update
        noise = np.random.normal(0, noise_scale, points.shape)
        points = points + alpha * grad + noise
        
        # Clip to disk
        new_radii = np.linalg.norm(points, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            points[mask] = points[mask] / new_radii[mask, np.newaxis]
            
        # Evaluate
        if iter_count % 20 == 0:
            score = _compute_hard_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        # Anneal
        beta *= 1.01
        alpha *= 0.995
        noise_scale *= 0.99
        
        # Safety: if beta is too high, reset? No, let it run.
        
    return best_points
