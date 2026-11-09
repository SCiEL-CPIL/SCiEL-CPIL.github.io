import numpy as np
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses a hexagonal lattice seed with gradient-based local optimization.
    """
    # Start with a hexagonal lattice configuration for 13 points
    # Center point + 6 inner ring + 6 outer ring (roughly)
    # Actually, for 13 points, a good layout is:
    # 1 center + 6 around it (hexagon) + 6 more around the outside
    # This gives a roughly hexagonal shape with 13 points
    
    s = 1.0  # spacing parameter
    sqrt3 = np.sqrt(3)
    
    # Center
    points = np.array([[0.0, 0.0]])
    
    # Inner hexagon (6 points)
    for k in range(6):
        angle = np.pi / 3 * k
        points = np.vstack([points, [s * np.cos(angle), s * np.sin(angle)]])
    
    # Outer hexagon (6 points) - offset to fill gaps
    # Place them at 1.5 * s distance, at angles offset by 30 degrees
    for k in range(6):
        angle = np.pi / 3 * k + np.pi / 6
        points = np.vstack([points, [1.5 * s * np.cos(angle), 1.5 * s * np.sin(angle)]])
    
    # Now optimize using a simple coordinate descent / local search
    # to maximize the minimum triangle area
    
    def min_triangle_area(pts):
        """Compute the minimum area among all triangles formed by 3 points."""
        n = len(pts)
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume is area
        except Exception:
            return float('inf')
    
    def score(pts):
        """The metric: A_min / A_hull."""
        a_min = min_triangle_area(pts)
        a_hull = hull_area(pts)
        if a_hull == 0 or np.isinf(a_hull):
            return 0.0
        return a_min / a_hull
    
    # Local optimization using random perturbations (hill climbing)
    best_points = points.copy()
    best_score = score(best_points)
    
    rng = np.random.default_rng(seed=12345)
    
    # Iterative local search
    step_size = 0.1
    for iteration in range(5000):
        # Try random perturbations
        improved = False
        for _ in range(20):
            candidate = best_points.copy()
            # Perturb a random point
            idx = rng.integers(0, 13)
            candidate[idx] += rng.normal(0, step_size, size=2)
            
            # Also try scaling
            if iteration % 100 == 0:
                candidate = candidate * (1.0 + rng.normal(0, 0.05))
            
            s_candidate = score(candidate)
            if s_candidate > best_score:
                best_points = candidate
                best_score = s_candidate
                improved = True
        
        # Reduce step size over time
        if iteration % 500 == 0 and iteration > 0:
            step_size *= 0.9
    
    # Final polish: try smaller steps
    step_size = 0.01
    for _ in range(2000):
        idx = rng.integers(0, 13)
        candidate = best_points.copy()
        candidate[idx] += rng.normal(0, step_size, size=2)
        s_candidate = score(candidate)
        if s_candidate > best_score:
            best_points = candidate
            best_score = s_candidate
    
    return best_points
