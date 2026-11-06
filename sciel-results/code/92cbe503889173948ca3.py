def optimize_sa(seed_points, max_iter=10000):
    """Simulated Annealing to maximize packing radius."""
    points = seed_points.copy()
    r = compute_radius(points)
    best_points = points.copy()
    best_r = r
    
    T = 0.005
    alpha = 0.9995
    step_size = 0.01
    
    for i in range(max_iter):
        if time.time() > DEADLINE - 20:
            break
            
        idx = np.random.randint(N)
        old_pos = points[idx].copy()
        points[idx] += np.random.randn(2) * step_size
        
        # Boundary check
        norm = np.linalg.norm(points[idx])
        if norm > 1.0:
            points[idx] *= 0.99 / norm
            
        new_r = compute_radius(points)
        
        if new_r >= r:
            r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            delta_r = r - new_r
            if np.random.rand() < np.exp(-delta_r / T):
                r = new_r
            else:
                points[idx] = old_pos
        
        T *= alpha
        
    return best_points

def polish(points, max_iter=500):
    """Fast local polish using gradient ascent on active constraints."""
    points = points.copy()
    r = compute_radius(points)
    best_points = points.copy()
    best_r = r
    
    step_size = 0.005
    
    for i in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        forces = np.zeros_like(points)
        
        for idx in range(N):
            dist_c = np.linalg.norm(points[idx])
            diffs = points[idx] - points
            dists = np.linalg.norm(diffs, axis=1)
            dists[idx] = np.inf
            min_dist, min_idx = np.min(dists), np.argmin(dists)
            
            # Determine active constraint
            # If boundary is tighter
            if (1.0 - dist_c) < (min_dist / 2.0):
                if dist_c > 0:
                    forces[idx] = -points[idx] / dist_c
            else:
                diff = points[idx] - points[min_idx]
                norm = np.linalg.norm(diff)
                if norm > 0:
                    forces[idx] = diff / norm
                else:
                    forces[idx] = np.random.randn(2)
                    
        points += step_size * forces
        
        # Boundary check
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] *= 0.99 / norms[mask]
            
        new_r = compute_radius(points)
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
            
    return best_points
