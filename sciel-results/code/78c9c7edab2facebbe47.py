import numpy as np

N = 14

def _min_triangle_area(points):
    """Compute the minimum area of all triangles formed by pairs of points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle formed by points i, j, k
                # 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    # Use scipy.spatial.ConvexHull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.volume # In 2D, volume is area

def _score(points):
    """Compute the normalized score: min triangle area / hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area == 0:
        return 0.0
    return min_area / hull_area

def construct():
    # Initialize with a regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_score = _score(points)
    
    # Local search parameters
    max_iterations = 5000
    step_size = 0.1
    rng = np.random.default_rng(42)
    
    for _ in range(max_iterations):
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Generate a random perturbation
        delta = rng.normal(0, step_size, size=2)
        
        new_points = points.copy()
        new_points[i] += delta
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            # Optionally reduce step size slightly to converge
            # step_size *= 0.999 
        else:
            # If rejected, we can try a smaller step or just continue
            pass
            
    return points
