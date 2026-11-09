# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    n = len(points)
    if n != 13:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Compute min triangle area (vectorized)
    # There are C(13,3) = 286 triangles
    min_area = float('inf')
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    
    if min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _initial_points():
    """Generate a structured initial configuration using Halton sequence."""
    n = 13
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, 2))
    for i in range(n):
        points[i, 0] = halton(i + 1, 2)
        points[i, 1] = halton(i + 1, 3)
    
    # Center and scale
    points = (points - 0.5) * 2.0
    return points


def _simulated_annealing(points, time_limit, rng):
    """Run simulated annealing to maximize the score."""
    best_points = points.copy()
    best_score = _compute_score(points)
    current_score = best_score
    
    T = 0.5  # Initial temperature
    T_min = 1e-8
    alpha = 0.995  # Cooling rate
    step_size = 0.3
    
    start_time = time.time()
    
    while T > T_min:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Generate perturbation
        new_points = best_points + rng.normal(0, step_size, size=(13, 2))
        
        new_score = _compute_score(new_points)
        
        # Accept if better or with probability exp((new - current) / T)
        if new_score >= current_score:
            current_score = new_score
            best_points = new_points.copy()
            best_score = max(best_score, new_score)
        else:
            delta = new_score - current_score
            if np.log(rng.random()) < delta / T:
                current_score = new_score
                best_points = new_points.copy()
        
        # Cool down
        T *= alpha
        # Reduce step size as we cool
        step_size *= 0.999
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    deadline = time.time() + 620  # Leave 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try multiple different initial configurations
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Start with Halton-based structured initial points
            points = _initial_points()
        else:
            # Perturb the best found so far or use random
            if best_points is not None:
                rng = np.random.default_rng(seed=1000 + attempt)
                points = best_points + rng.normal(0, 0.5, size=(13, 2))
            else:
                rng = np.random.default_rng(seed=42 + attempt)
                points = rng.random((13, 2)) - 0.5
        
        # Run simulated annealing with time budget
        time_left = deadline - time.time()
        if time_left <= 0:
            break
        
        time_per_run = time_left / max(1, 5 - attempt)
        rng = np.random.default_rng(seed=2000 + attempt)
        points, score = _simulated_annealing(points, time_per_run, rng)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Check if we're running out of time
        if time.time() > deadline - 30:
            break
    
    if best_points is None:
        # Fallback: return Halton points
        best_points = _initial_points()
    
    return best_points


# EVOLVE-BLOCK-END
