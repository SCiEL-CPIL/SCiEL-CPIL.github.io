"""Packing 200 equal circles in a unit square using force-directed relaxation."""
import numpy as np
import time

N = 200

def _score(points):
    """Compute the radius for a given set of points."""
    # Distance to walls
    wall_dist = np.minimum(0.5 - np.abs(points), axis=1)
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    r = min(wall_dist.min(), min_pair / 2.0)
    return r

def _initial_grid(n):
    """Create an initial grid layout."""
    m = int(np.ceil(np.sqrt(n)))
    # We need to place n points in an m x m grid, but only n of them
    # Use a dense grid and take the first n
    # Better: figure out the best rectangular grid
    # For n=200, sqrt is ~14.14, so 14x14=196, 15x14=210
    # Let's try different aspect ratios
    best_pts = None
    best_r = -1
    for m1 in range(10, 20):
        m2 = int(np.ceil(n / m1))
        if m1 * m2 < n:
            m2 += 1
        # Create m1 x m2 grid
        xs = (np.arange(m1) + 0.5) / m1 - 0.5
        ys = (np.arange(m2) + 0.5) / m2 - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        # Take first n points
        pts = grid[:n]
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    return best_pts

def _relax(points, max_iters=5000, deadline=None):
    """Force-directed relaxation to maximize minimum distance."""
    points = points.copy()
    n = len(points)
    
    # Parameters
    step_size = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances
        from scipy.spatial.distance import cdist
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        # Find minimum distance
        min_d = d.min()
        
        # Compute forces: repulsion from close neighbors, attraction to center
        forces = np.zeros_like(points)
        
        # Repulsive forces between close pairs
        # Only consider pairs within some cutoff for efficiency
        cutoff = min_d * 3
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < cutoff:
                    diff = points[i] - points[j]
                    dist = d[i, j]
                    if dist > 1e-10:
                        # Force proportional to 1/dist^2, direction away
                        f = diff / (dist * dist) * 0.1
                        forces[i] += f
                        forces[j] -= f
        
        # Wall constraints: push inward if too close to wall
        for i in range(n):
            for k in range(2):
                if points[i, k] < -0.5 + min_d / 2:
                    forces[i, k] += (min_d / 2 - (points[i, k] + 0.5)) * 10
                if points[i, k] > 0.5 - min_d / 2:
                    forces[i, k] -= (min_d / 2 - (0.5 - points[i, k])) * 10
        
        # Apply forces with step size
        # Normalize forces to avoid large jumps
        max_f = np.max(np.abs(forces))
        if max_f > 0:
            forces = forces / max_f * step_size
            points += forces
        
        # Project back into the square
        points = np.clip(points, -0.5, 0.5)
        
        # Cool down
        step_size *= cooling
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Initial grid
    points = _initial_grid(N)
    best_points = points.copy()
    best_r = _score(points)
    
    # Relaxation
    points = _relax(points, max_iters=2000, deadline=deadline)
    r = _score(points)
    if r > best_r:
        best_r = r
        best_points = points.copy()
    
    # Try a few more relaxation rounds if time permits
    if time.time() < deadline - 5:
        for restart in range(3):
            if time.time() > deadline:
                break
            # Perturb slightly
            noise = np.random.randn(N, 2) * 0.005
            perturbed = best_points + noise
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            points = _relax(perturbed, max_iters=1000, deadline=deadline)
            r = _score(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    # Final projection to ensure validity
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
