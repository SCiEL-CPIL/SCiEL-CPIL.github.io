import numpy as np
import time

N = 50

def _hex_init():
    """Initialize points on a hexagonal grid."""
    spacing = 0.25
    points = []
    k = 5
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) > N:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)
        points = points[idx[:N]]
    if len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
    # Scale to fit
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        points = points * (0.98 / max_r)
    return points

def _compute_bottleneck(points):
    """Returns (min_pair_dist, min_bound_dist, i, j) where (i,j) is the bottleneck pair or (i, -1) for boundary."""
    # Pairwise distances
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            dists[i, j] = d
            dists[j, i] = d
    
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    idx = np.argmin(dists)
    i, j = np.unravel_index(idx, dists.shape)
    
    # Boundary
    norms = np.linalg.norm(points, axis=1)
    max_norm = np.max(norms)
    min_bound_dist = 1.0 - max_norm
    k = np.argmax(norms)
    
    return min_pair_dist, min_bound_dist, i, j, k

def _optimize(points, max_time=10.0):
    start_time = time.time()
    step_size = 0.02
    points = points.copy()
    best_r = 0.0
    best_points = points.copy()
    
    while time.time() - start_time < max_time:
        min_pair_dist, min_bound_dist, i, j, k = _compute_bottleneck(points)
        
        current_r = min(min_pair_dist / 2.0, min_bound_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        if min_pair_dist < 2.0 * min_bound_dist:
            # Pair bottleneck
            if i == j: continue
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                noise_i = np.random.normal(0, 0.005, 2)
                noise_j = np.random.normal(0, 0.005, 2)
                points[i] += vec * step_size + noise_i
                points[j] -= vec * step_size + noise_j
        else:
            # Boundary bottleneck
            vec = -points[k] / np.linalg.norm(points[k])
            noise = np.random.normal(0, 0.005, 2)
            points[k] += vec * step_size + noise
            
        # Decay step size slightly
        step_size *= 0.99999
        
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_points = None
    best_r = 0.0
    
    points = _hex_init()
    
    while time.time() - start_time < deadline:
        if best_points is None:
            current_points = points + np.random.normal(0, 0.02, (N, 2))
        else:
            current_points = best_points + np.random.normal(0, 0.02, (N, 2))
            
        norms = np.linalg.norm(current_points, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            current_points[mask] = current_points[mask] * (0.99 / norms[mask])
            
        remaining = deadline - (time.time() - start_time)
        if remaining < 2.0:
            break
            
        opt_time = min(10.0, remaining * 0.5)
        new_points, new_r = _optimize(current_points, max_time=opt_time)
        
        if new_r > best_r:
            best_r = new_r
            best_points = new_points
            
    if best_points is None:
        best_points = _hex_init()
        
    return best_points
