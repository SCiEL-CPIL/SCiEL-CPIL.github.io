# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _vertex_score(dists, D):
    """
    Calculate the score for a single vertex given sorted distances and D.
    dists: array of 19 distances, sorted.
    D: the median of the closest gap.
    """
    # Check constraints
    if np.any(dists < 0.05):
        return 0.0
    if D < 0.1:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    # dists has length 19. Indices 0..18.
    # Gaps are d_{i+1} - d_i for i=0..17.
    gaps = np.diff(dists)
    k = np.argmin(gaps)
    
    # Determine indices for the 4 distances
    if k == 0:
        indices = [0, 1, 2, 3]
    elif k == 17:
        indices = [15, 16, 17, 18]
    else:
        indices = [k-1, k, k+1, k+2]
        
    selected = dists[indices]
    
    # Calculate raw score
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _evaluate_polygon(points):
    """
    Evaluate the polygon and return the minimum vertex score.
    points: (20, 2) array.
    """
    n = len(points)
    scores = []
    
    # Precompute all pairwise distances
    # dist[i, j] is distance between i and j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    
    for i in range(n):
        # Distances from i to all others, excluding self
        d_i = np.delete(dists[i], i)
        d_i = np.sort(d_i)
        
        # Find k
        gaps = np.diff(d_i)
        k = np.argmin(gaps)
        D = (d_i[k] + d_i[k+1]) / 2.0
        
        s = _vertex_score(d_i, D)
        scores.append(s)
        
    return np.min(scores)

def _polygon_from_params(params):
    """
    Generate polygon from Fourier coefficients.
    params: [c1, c2, c3]
    """
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    c1, c2, c3 = params
    
    r = 1.0 + c1 * np.cos(angles) + c2 * np.cos(2*angles) + c3 * np.cos(3*angles)
    
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Check convexity
    # Calculate cross products of consecutive edges
    # v_i = p_{i+1} - p_i
    # cross = v_i x v_{i+1}
    # For CCW convex, cross > 0
    
    v = np.roll(points, -1, axis=0) - points
    cross = v[:, 0] * np.roll(v, -1, axis=0)[:, 1] - v[:, 1] * np.roll(v, -1, axis=0)[:, 0]
    
    if np.any(cross <= 1e-6):
        return None
        
    return points

def _objective(params):
    """
    Objective function for optimizer.
    Returns negative score (to maximize).
    """
    points = _polygon_from_params(params)
    if points is None:
        return 1.0 # Penalty for non-convex
        
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    mean_dist = np.mean(np.linalg.norm(points, axis=1))
    if mean_dist <= 0.01:
        return 1.0
    points = points / mean_dist
    
    score = _evaluate_polygon(points)
    
    # If score is 0, penalize heavily
    if score == 0.0:
        return 1.0
        
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Bounds for coefficients
    # Small coefficients ensure convexity
    bounds = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
    
    # Initial best
    best_score = -1.0
    best_points = None
    
    # Regular polygon as fallback
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_x = np.cos(angles)
    reg_y = np.sin(angles)
    reg_points = np.column_stack((reg_x, reg_y))
    reg_centroid = np.mean(reg_points, axis=0)
    reg_points = reg_points - reg_centroid
    reg_mean_dist = np.mean(np.linalg.norm(reg_points, axis=1))
    reg_points = reg_points / reg_mean_dist
    best_points = reg_points
    best_score = _evaluate_polygon(reg_points)
    
    # Differential Evolution
    # Use a moderate population size and max iterations
    # We need to be fast.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42
        )
        
        if time.time() > deadline - 10:
            # Return best found so far
            pass
        else:
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = _polygon_from_params(result.x)
                
                if best_points is not None:
                    # Normalize
                    centroid = np.mean(best_points, axis=0)
                    best_points = best_points - centroid
                    mean_dist = np.mean(np.linalg.norm(best_points, axis=1))
                    if mean_dist > 0.01:
                        best_points = best_points / mean_dist
                    else:
                        best_points = reg_points
                        best_score = _evaluate_polygon(reg_points)
                        
    except Exception:
        pass

    # Final check
    if best_points is None:
        best_points = reg_points
        
    # Ensure the returned points are normalized as per task description
    # "subtract the centroid, divide by the mean distance to it"
    centroid = np.mean(best_points, axis=0)
    best_points = best_points - centroid
    mean_dist = np.mean(np.linalg.norm(best_points, axis=1))
    if mean_dist > 0.01:
        best_points = best_points / mean_dist
    else:
        # Fallback to regular if normalization fails
        best_points = reg_points

    return best_points

# EVOLVE-BLOCK-END
