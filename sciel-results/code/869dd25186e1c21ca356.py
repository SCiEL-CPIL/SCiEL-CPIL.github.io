import numpy as np
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses a structured approach: start with a regular polygon + center,
    then optimize using gradient-free methods with symmetry exploitation.
    """
    
    def compute_min_triangle_area(pts):
        """Vectorized computation of minimum triangle area."""
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
    
    def compute_hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return float('inf')
    
    def compute_score(pts):
        """Compute A_min / A_hull."""
        a_min = compute_min_triangle_area(pts)
        a_hull = compute_hull_area(pts)
        if a_hull <= 0 or np.isinf(a_hull):
            return 0.0
        return a_min / a_hull
    
    # Strategy: Use a regular 12-gon with center point
    # This has high symmetry and is a known good starting point for Heilbronn problems
    
    # Regular 12-gon vertices + center
    n_sides = 12
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    radius = 1.0
    
    points = np.array([[0.0, 0.0]])  # Center
    for angle in angles:
        points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
    
    # Optimize the radius and individual point positions
    # Use a multi-start approach with different initial configurations
    
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: Regular 12-gon + center (current)
    configs.append(points.copy())
    
    # Config 2: Two concentric hexagons (6+6) + center
    points2 = np.array([[0.0, 0.0]])
    for k in range(6):
        angle = np.pi / 3 * k
        points2 = np.vstack([points2, [0.8 * np.cos(angle), 0.8 * np.sin(angle)]])
    for k in range(6):
        angle = np.pi / 3 * k + np.pi / 6
        points2 = np.vstack([points2, [1.2 * np.cos(angle), 1.2 * np.sin(angle)]])
    configs.append(points2.copy())
    
    # Config 3: Regular 13-gon
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    points3 = np.array([[1.0 * np.cos(a), 1.0 * np.sin(a)] for a in angles13])
    configs.append(points3.copy())
    
    # Config 4: Hexagonal lattice arrangement
    sqrt3 = np.sqrt(3)
    s = 1.0
    points4 = np.array([
        [0, 0],
        [s, 0],
        [-s, 0],
        [0, s*sqrt3/2],
        [0, -s*sqrt3/2],
        [s/2, s*sqrt3/2],
        [-s/2, s*sqrt3/2],
        [s/2, -s*sqrt3/2],
        [-s/2, -s*sqrt3/2],
        [1.5*s, s*sqrt3/2],
        [-1.5*s, s*sqrt3/2],
        [1.5*s, -s*sqrt3/2],
        [-1.5*s, -s*sqrt3/2]
    ], dtype=float)
    configs.append(points4.copy())
    
    # Optimize each configuration
    for init_points in configs:
        current_points = init_points.copy()
        current_score = compute_score(current_points)
        
        if current_score > best_score:
            best_points = current_points.copy()
            best_score = current_score
        
        # Local optimization with adaptive step size
        rng = np.random.default_rng(seed=42)
        step_size = 0.05
        
        for iteration in range(3000):
            improved = False
            for _ in range(10):
                candidate = current_points.copy()
                idx = rng.integers(0, 13)
                candidate[idx] += rng.normal(0, step_size, size=2)
                
                cand_score = compute_score(candidate)
                if cand_score > current_score:
                    current_points = candidate
                    current_score = cand_score
                    improved = True
            
            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    break
            
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
    
    # Final fine-tuning
    step_size = 0.01
    rng = np.random.default_rng(seed=999)
    for _ in range(2000):
        idx = rng.integers(0, 13)
        candidate = best_points.copy()
        candidate[idx] += rng.normal(0, step_size, size=2)
        cand_score = compute_score(candidate)
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
    
    return best_points
