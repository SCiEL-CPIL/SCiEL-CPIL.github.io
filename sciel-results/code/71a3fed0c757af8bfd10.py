"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0.0 for degenerate or invalid inputs.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(n), 3):
        # Area of triangle with vertices points[i], points[j], points[k]
        # = 0.5 * |cross(pj - pi, pk - pi)|
        pi = points[i]
        pj = points[j]
        pk = points[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _heuristic_init():
    """
    Generate an initial configuration using a structured heuristic.
    We place points on a circle with some perturbation, or use a grid-based approach.
    A good starting point for maximizing minimum triangle area is often a regular polygon
    or points spread out on a circle.
    """
    # Try a regular 13-gon on a unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # The convex hull of a regular polygon is the polygon itself.
    # Area of regular n-gon with circumradius R=1: A = 0.5 * n * sin(2*pi/n)
    # For n=13: A = 0.5 * 13 * sin(2*pi/13)
    # Min triangle area for regular polygon: smallest triangles are formed by adjacent vertices.
    # This is a decent baseline. Let's also try a few other structured inits.
    
    # Alternative: Points on two concentric circles?
    # Or a grid?
    # Let's stick with the regular polygon as a strong baseline.
    return points

def _local_search(points, max_iter=5000, step_size=0.1):
    """
    Perform a local search to maximize the score.
    Uses a simple hill-climbing approach with random perturbations.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Precompute all triangle indices for speed
    n = len(current_points)
    triangles = list(itertools.combinations(range(n), 3))
    
    for it in range(max_iter):
        # Pick a random point to move
        idx = np.random.randint(n)
        original_pos = current_points[idx].copy()
        
        # Generate a random perturbation
        # Use a Gaussian perturbation
        delta = np.random.normal(0, step_size, 2)
        new_pos = original_pos + delta
        current_points[idx] = new_pos
        
        new_score = _compute_score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Revert
            current_points[idx] = original_pos
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration
    best_points = _heuristic_init()
    best_score = _compute_score(best_points)
    
    # Perform local search
    # We can do multiple restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            current_points = best_points.copy()
        else:
            # Add some noise to the best solution for restart
            current_points = best_points + np.random.normal(0, 0.1, (N, 2))
        
        # Adjust step size: start larger, decrease over time
        step_size = 0.2 / (1 + r * 0.2)
        
        # Run local search
        new_points, new_score = _local_search(current_points, max_iter=3000, step_size=step_size)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_points
# EVOLVE-BLOCK-END
