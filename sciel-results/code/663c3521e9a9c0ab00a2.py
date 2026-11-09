import numpy as np
import time

N = 14

def min_triangle_area(points):
    # points: (N, 2)
    # Compute area of all triples
    # Using cross product: 0.5 * | (p1-p0) x (p2-p0) |
    # Vectorized:
    # i, j, k indices
    # This might be memory heavy if we do all at once, but N=14 is small.
    # Number of triangles = 364.
    # We can just loop or use simple vectorization.
    
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p0 = points[i]
                p1 = points[j]
                p2 = points[k]
                # Cross product in 2D
                area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
                areas.append(area)
    return np.min(areas)

def convex_hull_area(points):
    # For a convex polygon given in order, area is easy.
    # But points might not be in order if we jitter.
    # However, if we start with regular polygon and jitter slightly, they remain in order.
    # To be safe, we should compute convex hull area.
    # Since the task says "normalized by the point set convex hull area", and the points are likely to remain in convex position (or close to it), 
    # if we ensure they stay in convex position, the hull area is the polygon area.
    # If a point moves inside, the hull area changes.
    # Let's use scipy.spatial.ConvexHull for robustness.
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # Fallback if points are collinear or degenerate (unlikely)
        return 1.0

def construct():
    # Start with regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack((np.cos(angles), np.sin(angles))).astype(float)
    
    current_score = min_triangle_area(points) / convex_hull_area(points)
    best_points = points.copy()
    best_score = current_score
    
    start_time = time.time()
    timeout = 110 # Keep 10s buffer
    
    step_size = 0.05
    iterations = 0
    
    while time.time() - start_time < timeout:
        iterations += 1
        # Pick a random point to perturb
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Random perturbation
        noise = np.random.normal(0, step_size, 2)
        points[idx] += noise
        
        new_min_area = min_triangle_area(points)
        new_hull_area = convex_hull_area(points)
        new_score = new_min_area / new_hull_area
        
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
        else:
            # Revert if not better than best, or just revert to previous state?
            # Standard hill climbing: accept if better than current, reject otherwise.
            # But we track best. Let's do simple hill climbing on current state.
            points[idx] = original_point
            
        # Adapt step size?
        # If we haven't improved in a while, reduce step size.
        # Simple heuristic: reduce step size every 1000 iterations
        if iterations % 1000 == 0:
            step_size *= 0.95

    return best_points
