"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Calculate the score: min triangle area / hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-9:
        return 0.0
    min_tri_a = _min_triangle_area(points)
    return min_tri_a / hull_a


def construct():
    # Initial configuration: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles))).copy()
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    iterations = 50000
    
    current_points = points.copy()
    current_score = best_score
    
    for i in range(iterations):
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        
        # Save original position
        orig = current_points[idx].copy()
        
        # Perturb
        noise = np.random.normal(0, T, 2)
        current_points[idx] += noise
        
        # Calculate new score
        new_score = _score(current_points)
        
        # Acceptance criteria
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Reject move
            current_points[idx] = orig
            
        # Cool down
        T *= cooling_rate
        
    return best_points
# EVOLVE-BLOCK-END
