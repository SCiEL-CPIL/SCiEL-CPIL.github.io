import numpy as np
from scipy.spatial import ConvexHull

N = 50

def construct():
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Repulsive Force Optimization (Gradient Descent on 1/d^2)
    step_size = 0.01
    for _ in range(500):
        dist2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * points @ points.T
        np.fill_diagonal(dist2, 1.0)
        dist4 = dist2**2
        weights = 1.0 / dist4
        np.fill_diagonal(weights, 0.0)
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum_points = weights.T @ points
        
        forces = points * sum_weights[:, None] - weighted_sum_points
        
        points += step_size * forces
        points /= np.linalg.norm(points, axis=1)[:, None]
        step_size *= 0.99

    # 3. Local Search for Covering Radius Minimization
    max_iter = 3000
    base_step = 0.2
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            points += np.random.normal(0, 0.001, (N, 3))
            points /= np.linalg.norm(points, axis=1)[:, None]
            continue

        min_m = 1.0
        best_y = None
        best_facet = None

        for facet in facets:
            p1 = points[facet[0]]
            p2 = points[facet[1]]
            p3 = points[facet[2]]
            
            v = np.cross(p2 - p1, p3 - p1)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            y = v / norm_v
            
            # Ensure y is on the correct side (outward)
            c = (p1 + p2 + p3) / 3.0
            if np.dot(c, y) < 0:
                y = -y
            
            dots = points @ y
            m = np.max(dots)
            
            if m < min_m:
                min_m = m
                best_y = y
                best_facet = facet

        if best_y is None:
            break

        # Decay step size
        current_step = base_step * (0.999 ** it)
        
        # Move vertices of best_facet towards best_y
        for idx in best_facet:
            diff = best_y - points[idx]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                points[idx] += current_step * diff / norm_diff
                
        # Normalize
        points /= np.linalg.norm(points, axis=1)[:, None]
        
        # Random restart/jitter to escape local minima
        if it % 100 == 0 and it > 0:
            # Small jitter
            points += np.random.normal(0, 0.005, (N, 3))
            points /= np.linalg.norm(points, axis=1)[:, None]

    return points
