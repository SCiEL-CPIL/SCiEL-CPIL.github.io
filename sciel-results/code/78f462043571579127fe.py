import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Handle zero norms to avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius_and_worst_facet(X):
    """
    Computes the exact covering radius (max circumradius of hull facets)
    and returns the radius, the worst simplex indices, and the circumcenter.
    """
    try:
        hull = ConvexHull(X)
    except Exception:
        # If hull fails (e.g., points are coplanar or degenerate), return a bad score
        return np.pi, None, None

    max_radius = 0.0
    worst_simplex = None
    worst_center = None
    
    # Iterate over all triangular facets
    for simplex in hull.simplices:
        v1 = X[simplex[0]]
        v2 = X[simplex[1]]
        v3 = X[simplex[2]]
        
        # Compute normal vector
        edge1 = v2 - v1
        edge2 = v3 - v1
        n = np.cross(edge1, edge2)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-10:
            continue
            
        # Circumcenter direction is along the normal
        c = n / norm_n
        
        # Ensure c points outwards (same direction as v1 from origin)
        if np.dot(c, v1) < 0:
            c = -c
            
        # Calculate angular distance from c to v1
        # d = ||c - v1||
        # cos(theta) = (c dot v1) since both are unit vectors
        # Actually, easier: angle = arccos(c dot v1)
        # Or using distance: sin(theta/2) = ||c - v1|| / 2
        # theta = 2 * arcsin(||c - v1|| / 2)
        
        dist = np.linalg.norm(c - v1)
        # Clip to avoid domain errors in arcsin
        sin_half_theta = np.clip(dist / 2.0, 0.0, 1.0)
        radius = 2.0 * np.arcsin(sin_half_theta)
        
        if radius > max_radius:
            max_radius = radius
            worst_simplex = simplex.copy()
            worst_center = c
            
    if worst_simplex is None:
        # Fallback if no valid facets found
        return np.pi, None, None
        
    return max_radius, worst_simplex, worst_center

def _local_refine(X, deadline, max_iters=100, initial_step=0.1):
    """
    Refines the configuration by moving vertices of the worst facet
    towards its circumcenter.
    """
    current = X.copy()
    step = initial_step
    best_radius = _compute_covering_radius_and_worst_facet(current)[0]
    
    for _ in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        radius, worst_simplex, worst_center = _compute_covering_radius_and_worst_facet(current)
        
        if worst_simplex is None or worst_center is None:
            break
            
        # If radius is already very small, we can stop
        if radius < 1e-6:
            break
            
        # Move vertices of the worst simplex towards the circumcenter
        improved = False
        for idx in worst_simplex:
            v = current[idx]
            # Direction to move: from v to worst_center
            dir_vec = worst_center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir < 1e-10:
                continue
            dir_vec /= norm_dir
            
            # Update point
            new_v = v + step * dir_vec
            current[idx] = new_v / np.linalg.norm(new_v)
            improved = True
            
        if not improved:
            break
            
        # Evaluate new radius
        new_radius, _, _ = _compute_covering_radius_and_worst_facet(current)
        
        # Accept if better
        if new_radius < best_radius:
            best_radius = new_radius
            # Decay step size slightly if we improved
            step *= 0.95
        else:
            # Revert if not better
            current = X.copy()
            # Try smaller step
            step *= 0.8
            if step < 1e-4:
                break
                
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Fibonacci spiral
    best_X = _fibonacci_spiral(N)
    best_X = _normalize(best_X)
    
    # Initial evaluation
    best_val, _, _ = _compute_covering_radius_and_worst_facet(best_X)
    
    # Basin Hopping Loop
    # We maintain a single "current" state that is perturbed and refined.
    # If it doesn't improve, we try different perturbation strategies.
    
    current_X = best_X.copy()
    perturbation_scale = 0.1
    
    # To avoid getting stuck in a very deep local minimum immediately,
    # we can occasionally restart from a random rotation of the best found so far.
    
    while time.time() < deadline - 5.0:
        # 1. Perturb
        # Random noise
        noise = np.random.normal(0, perturbation_scale, (N, 3))
        perturbed_X = current_X + noise
        perturbed_X = _normalize(perturbed_X)
        
        # 2. Local Refine
        refined_X = _local_refine(perturbed_X, deadline, max_iters=50, initial_step=0.05)
        
        # 3. Evaluate
        radius, _, _ = _compute_covering_radius_and_worst_facet(refined_X)
        
        # 4. Update Best
        if radius < best_val:
            best_val = radius
            best_X = refined_X.copy()
            # Reset perturbation scale to be larger to explore more? 
            # Or keep it adaptive.
            perturbation_scale = max(0.01, perturbation_scale * 0.9)
        else:
            # If no improvement, increase perturbation to escape
            perturbation_scale = min(0.5, perturbation_scale * 1.1)
            
        # 5. Update Current State
        # Keep the better of (current, refined) to maintain a good baseline for next perturbation
        # Actually, standard basin hopping keeps the current state as the start for the next perturbation.
        # If we always go back to best_X, we might explore the same basin repeatedly.
        # Let's keep a "replica" that is the best found so far, but perturb from a slightly worse 
        # state to allow drift, OR just perturb from the best and rely on noise.
        # Given the short time, perturbing from the global best is usually safest to ensure
        # we are always trying to improve the global best directly.
        
        # However, if we always perturb from best_X, and local refine is greedy, we might
        # converge too quickly to a local min. 
        # Let's introduce a "current" state that is not necessarily the global best,
        # but is the result of the last iteration.
        
        # Logic:
        # current_X starts as best_X.
        # Perturb current_X -> perturbed.
        # Refine perturbed -> refined.
        # If refined < best_val: best_val = refined, best_X = refined.
        # Set current_X = refined (regardless of improvement) to allow drift away from best?
        # No, that risks losing the best.
        # Standard Basin Hopping:
        # 1. Perturb current
        # 2. Minimize
        # 3. If new_min < current_min: current = new_min
        # 4. If new_min < global_min: global = new_min
        # Here, let's do:
        # current_X = refined (if refined is not too bad)
        # But if refined is worse than current_X, we should revert current_X?
        
        # Let's simplify:
        # We always perturb the GLOBAL BEST. This is "Global Best" strategy.
        # It is slower to explore new basins but very effective at polishing.
        # To explore, we increase perturbation_scale.
        
        # Let's try a hybrid:
        # Keep `current_X` as the state we are perturbing.
        # If `refined` is better than `current_X`, update `current_X = refined`.
        # If `refined` is better than `best_X`, update `best_X = refined`.
        # This allows the search to drift away from the global best if it finds a slightly worse
        # but potentially more promising basin (simulated annealing-like).
        
        cur_val, _, _ = _compute_covering_radius_and_worst_facet(current_X)
        if radius < cur_val:
            current_X = refined_X.copy()
        else:
            # Revert to previous current state if this path was worse
            # This ensures we don't drift into a bad region
            pass # current_X stays as is
            
        # Occasionally, if we are stuck (no improvement for a while), 
        # reset current_X to best_X to focus on polishing again.
        # For simplicity, we just rely on the perturbation scale adaptation.

    return best_X
