"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: shoelace formula on convex hull vertices
        from scipy.spatial import ConvexHull as CH
        try:
            h = CH(points)
            return h.volume
        except Exception:
            return 1.0  # Default to avoid division by zero


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross product|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Use a regular polygon with 14 vertices as the initial configuration,
    then apply local optimization (hill climbing with small perturbations) to improve
    the score.
    """
    # Start with a regular 14-gon inscribed in the unit circle
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    current_score = _score(points)
    
    # Local optimization: try small perturbations to each point
    rng = np.random.RandomState(42)
    best_points = points.copy()
    best_score = current_score
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        if restart == 0:
            # Already started with regular polygon
            pass
        elif restart == 1:
            # Regular polygon with slight rotation
            angles = 2 * np.pi * np.arange(N) / N + 0.1
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 2:
            # Two concentric rings: 7 points on outer, 7 on inner
            outer_angles = 2 * np.pi * np.arange(7) / 7
            inner_angles = 2 * np.pi * np.arange(7) / 7 + np.pi / 7
            outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            points = np.vstack([outer, inner])
        elif restart == 3:
            # 3 rings: 5, 5, 4 points
            r1_angles = 2 * np.pi * np.arange(5) / 5
            r2_angles = 2 * np.pi * np.arange(5) / 5 + np.pi / 5
            r3_angles = 2 * np.pi * np.arange(4) / 4
            r1 = np.column_stack([np.cos(r1_angles), np.sin(r1_angles)])
            r2 = 0.6 * np.column_stack([np.cos(r2_angles), np.sin(r2_angles)])
            r3 = 0.3 * np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
            points = np.vstack([r1, r2, r3])
        else:
            # Random initialization
            points = rng.rand(N, 2)
        
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Hill climbing: iteratively perturb points
        for iteration in range(500):
            improved = False
            for i in range(N):
                # Try perturbing point i in several directions
                for scale in [0.01, 0.005, 0.001]:
                    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        dx = scale * np.cos(angle)
                        dy = scale * np.sin(angle)
                        new_points = points.copy()
                        new_points[i] = points[i] + np.array([dx, dy])
                        
                        new_score = _score(new_points)
                        if new_score > current_score:
                            points = new_points
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = points.copy()
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    return best_points
# EVOLVE-BLOCK-END
