import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    # Vectorized approach for speed
    # i, j, k indices
    min_area = float('inf')
    # There are C(13,3) = 286 triangles
    # Loop is fast enough, but let's try to be efficient
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _fast_score(points):
    """
    Optimized scoring for the inner loop of Simulated Annealing.
    Avoids overhead of ConvexHull if possible? No, hull area changes.
    But we can pre-compute indices.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area = float('inf')
    # Pre-computed indices for 13 points
    # Using a static list of tuples for speed
    for i, j, k in _TRIS:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if area < min_area:
            min_area = area
            if min_area < 1e-12:
                return 0.0 # Early exit if nearly collinear

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

# Pre-compute triangle indices for 13 points
_TRIS = [(i, j, k) for i in range(13) for j in range(i+1, 13) for k in range(j+1, 13)]

def heilbronn_convex13() -> np.ndarray:
    """
    Use Simulated Annealing to find a configuration of 13 points that maximizes
    the ratio of the minimum triangle area to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initialize with a regular polygon (good starting geometry)
    # Or a few diverse starts. Let's do a regular polygon centered at 0.
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    current_points = init_points.copy()
    current_score = _score(current_points)
    
    if current_score > best_score:
        best_score = current_score
        best_points = current_points.copy()

    # Simulated Annealing Parameters
    # We need to balance exploration (high T) and exploitation (low T)
    # The score is roughly in [0, 0.1] for 13 points? 
    # Actually, for n=13, the Heilbronn number is known to be around 0.01-0.02? 
    # Wait, the seed score is 54.6? 
    # Let's check the seed score logic. 
    # If seed score is 54.6, that's huge. 
    # A_min / A_hull. 
    # If points are in a unit square, A_hull ~ 1. 
    # Min triangle area for 13 points in unit square... 
    # Maybe the seed is scaling? 
    # No, the seed returns points. 
    # Let's assume the score magnitude is correct. 
    # If score is ~50, then T should start around 50 and decay.
    
    T_start = 100.0
    T_end = 1e-5
    cooling_rate = 0.999
    
    # Step size for moves
    step_size = 0.1
    
    # We will run SA for a fixed number of iterations or until time runs out.
    # Each iteration: 
    # 1. Pick a random point
    # 2. Perturb it by Gaussian noise
    # 3. Calculate new score
    # 4. Accept if better, or with prob exp((new-old)/T)
    
    # To speed up, we can try to optimize the score calculation.
    # But 286 triangles is small.
    
    # Let's run multiple restarts if time permits.
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_points = init_points.copy()
            current_score = _score(current_points)
        else:
            # Random restart: jitter the best found so far, or random
            # Jittering best is often better
            rng = np.random.default_rng(restart)
            current_points = best_points + rng.normal(0, 0.1, (13, 2))
            current_score = _score(current_points)
            
        T = T_start
        
        # Inner loop
        # Estimate iterations per second. 
        # Score calculation is the bottleneck.
        # Let's run for a fixed number of steps per restart, say 50000.
        # 50000 * 5 restarts = 250,000 evaluations.
        # If each eval is 1ms, that's 250s. Fits in budget.
        
        max_iters_per_restart = 50000
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline:
                break
                
            # Move
            idx = np.random.randint(13)
            # Gaussian perturbation
            # Scale step size with T? Or keep constant?
            # Constant step size is often better for geometry, but T controls acceptance.
            noise = np.random.normal(0, step_size, 2)
            
            old_point = current_points[idx].copy()
            new_point = old_point + noise
            current_points[idx] = new_point
            
            new_score = _score(current_points)
            
            # Acceptance criterion
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Probabilistic acceptance
                # delta = new - old (negative)
                # prob = exp(delta / T)
                # If T is very small, prob is 0.
                try:
                    prob = np.exp((new_score - current_score) / T)
                    if np.random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        current_points[idx] = old_point
                except OverflowError:
                    # Revert
                    current_points[idx] = old_point
            
            # Cooling
            T *= cooling_rate
            
            # If T is very low, we can stop this restart early or reduce step size
            if T < T_end:
                break

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))

    return best_points
