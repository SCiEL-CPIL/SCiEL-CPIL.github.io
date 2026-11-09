# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist


def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    
    # Check for repeated points (consecutive)
    # The task says "no repeated points", usually implies distinct vertices.
    # In a convex polygon with sorted angles, consecutive points are distinct if radius > 0 and angle gap > 0.
    # We should check all pairs? "No repeated points" usually means $p_i \neq p_j$ for $i \neq j$.
    # With sorted angles and positive radii, this is guaranteed unless angle gaps are 0 or radii are 0.
    # Let's check min distance between any two points to be safe, or just rely on angle sorting.
    # Angle sorting ensures cyclic order.
    
    # Check turn directions
    # v1 = p_{i+1} - p_i, v2 = p_{i+2} - p_{i+1}
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    p0 = polygon
    p1 = np.roll(polygon, -1, axis=0)
    p2 = np.roll(polygon, -2, axis=0)
    
    v1 = p1 - p0
    v2 = p2 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # All crosses must be non-zero and same sign
    if np.any(np.abs(cross) < 1e-10):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    # Check total turning
    # For a strictly convex polygon, total turning is 2*pi.
    # We can verify this by checking if the polygon is simple and convex, which the cross product check does.
    # Is it possible to have same sign crosses but not 2*pi turning?
    # If it's a simple polygon, yes, it's 2*pi.
    # Does same sign crosses imply simple?
    # For a polygon defined by sorted angles and positive radii, it is star-shaped with respect to the origin, hence simple.
    # So the cross product check is sufficient for convexity in this parameterization.
    
    return True


def _score_polygon_fast(polygon):
    """
    Compute the metric score for a polygon. Returns 0 if invalid.
    Vectorized as much as possible.
    """
    if polygon is None:
        return 0.0
        
    n = len(polygon)
    
    # Normalize
    norm_poly = _normalize(polygon)
    if norm_poly is None:
        return 0.0
        
    # Check convexity
    if not _is_valid_convex(norm_poly):
        return 0.0
        
    # Compute distance matrix (n x n)
    # d[i, j] = || p_i - p_j ||
    diff = norm_poly[:, np.newaxis, :] - norm_poly[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(dists, np.inf)
    
    min_scores = 1.0
    min_d_found = 0.0 # To check for early exit if any vertex is bad? No, we need min score.
    
    for i in range(n):
        d_sorted = np.sort(dists[i])
        
        # Check for any distance < 0.05
        if d_sorted[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1})/2
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Determine the 4 distances to use
        if k == 0:
            ds = d_sorted[0:4]
        elif k == 17: # n-2 is 18, so max k is 17? 
            # d_sorted has 19 elements (0..18).
            # gaps has 18 elements (0..17).
            # So k is in 0..17.
            # If k=17, we use d_15..d_18.
            ds = d_sorted[15:19]
        else:
            ds = d_sorted[k-1:k+3]
            
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(ds / D, D / ds)
        raw = np.mean(ratios)
        
        # vertex score = min(1, 1/raw)
        score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_scores:
            min_scores = score
            
    return min_scores


def _polygon_from_params(params):
    """
    params: (40,) array.
    First 20: radii.
    Next 20: angles (unsorted).
    Returns: (20, 2) polygon.
    """
    n = 20
    radii = params[:n]
    angles = params[n:]
    
    # Ensure radii are positive
    radii = np.abs(radii) + 1e-6
    
    # Sort angles to ensure cyclic order
    angles_sorted = np.sort(angles)
    
    x = radii * np.cos(angles_sorted)
    y = radii * np.sin(angles_sorted)
    
    return np.stack([x, y], axis=1)


def _objective(params):
    """
    Objective function for minimization.
    Returns negative score.
    """
    polygon = _polygon_from_params(params)
    score = _score_polygon_fast(polygon)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2)
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Initial guess: regular polygon
    init_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_radii = np.ones(n)
    init_params = np.concatenate([init_radii, init_angles])
    
    # Bounds for DE
    # Radii: 0.5 to 1.5 (normalized later, so scale doesn't matter much, but relative scale does)
    # Angles: 0 to 2*pi
    bounds = [(0.5, 1.5)] * n + [(0, 2*np.pi)] * n
    
    # Run Differential Evolution
    # Popsize 10-15 is usually good. 
    # Maxiter 100-200.
    # Tolerance for stopping.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        best_params = result.x
        best_score = -result.fun
        
    except Exception:
        # Fallback to initial
        best_params = init_params
        best_score = _score_polygon_fast(_polygon_from_params(init_params))
        
    # Polish with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            # NM is local, no bounds needed strictly, but we can constrain if needed.
            # It handles the non-smoothness reasonably well if initialized close.
            res = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
            )
            
            if res.fun < _objective(best_params):
                best_params = res.x
                best_score = -res.fun
                
        except Exception:
            pass
            
    # Final check and return
    best_polygon = _polygon_from_params(best_params)
    
    # Ensure the returned polygon is valid and normalized?
    # The task says "returning a (20, 2) array of finite vertex coordinates".
    # The evaluator normalizes it.
    # We should return the polygon that yields the best score.
    
    return best_polygon


# EVOLVE-BLOCK-END
