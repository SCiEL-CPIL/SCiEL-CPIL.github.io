# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist

def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be +-2*pi
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        # Normalize to [-pi, pi]
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True

def _vertex_score(vertex, polygon):
    """Compute the score for a single vertex."""
    n = len(polygon)
    # Distances to all other vertices
    diffs = polygon - vertex
    dists = np.linalg.norm(diffs, axis=1)
    # Set self distance to inf
    idx = np.argmin(dists)
    dists[idx] = np.inf
    # Sort the 19 distances
    dists_sorted = np.sort(dists)
    # Check if any distance < 0.05
    if np.any(dists_sorted < 0.05):
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = dists_sorted[1:] - dists_sorted[:-1]
    k = np.argmin(gaps)
    D = (dists_sorted[k] + dists_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # Select four distances
    if k == 0:
        four_dists = dists_sorted[0:4]
    elif k == 17:
        four_dists = dists_sorted[15:19]
    else:
        four_dists = dists_sorted[k - 1:k + 3]
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(four_dists / D, D / four_dists)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _polygon_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        scores.append(_vertex_score(polygon[i], polygon))
    return min(scores)

def _smooth_vertex_score(vertex, polygon, beta=20.0):
    """
    Compute a smooth approximation of the vertex score.
    Uses log-sum-exp to approximate max(d/D, D/d).
    """
    n = len(polygon)
    diffs = polygon - vertex
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    dists[idx] = np.inf
    dists_sorted = np.sort(dists)
    
    min_dist = dists_sorted[0]
    if min_dist < 0.05:
        return 0.0 

    gaps = dists_sorted[1:] - dists_sorted[:-1]
    k = np.argmin(gaps)
    D = (dists_sorted[k] + dists_sorted[k + 1]) / 2
    
    if D < 0.1:
        return 0.0

    if k == 0:
        four_dists = dists_sorted[0:4]
    elif k == 17:
        four_dists = dists_sorted[15:19]
    else:
        four_dists = dists_sorted[k - 1:k + 3]
    
    ratios_a = four_dists / D
    ratios_b = D / four_dists
    
    diff = np.abs(ratios_a - ratios_b)
    soft_abs = (1.0 / beta) * np.log1p(np.exp(2.0 * beta * diff))
    smooth_max = (ratios_a + ratios_b + soft_abs) / 2.0
    
    raw = np.mean(smooth_max)
    
    return 1.0 / raw

def _smooth_polygon_score(polygon, beta_min=20.0, beta_max=20.0):
    """
    Compute the soft-min of the 20 smooth vertex scores.
    soft_min(x_1, ..., x_n) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    scores = np.array([_smooth_vertex_score(polygon[i], polygon, beta_max) for i in range(len(polygon))])
    if np.any(scores < 1e-9):
        return 0.0
    
    max_score = np.max(scores)
    shifted = -beta_min * (scores - max_score)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = max_score - (1.0 / beta_min) * log_sum_exp
    
    return soft_min

def _score_from_radii(radii):
    """Compute score given radii for fixed angles."""
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score(polygon)

def _optimize_radii_smooth(init_radii, start_time, deadline):
    """Optimize radii using L-BFGS-B on a smooth surrogate."""
    n = len(init_radii)
    
    def objective(radii):
        if time.time() > deadline - 5.0:
            return 1e9
        radii = np.maximum(radii, 0.1)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        polygon = _normalize(polygon)
        
        if not _is_valid_convex(polygon):
            return 1e9
        
        score = _smooth_polygon_score(polygon)
        return -score
    
    x0 = init_radii
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
    except Exception:
        return init_radii
    
    if time.time() > deadline - 5.0:
        return init_radii
    
    best_r = res.x
    best_score = -res.fun
    init_score = -objective(x0)
    
    exact_init_score = _score_from_radii(x0)
    exact_best_score = _score_from_radii(best_r)
    
    if exact_best_score > exact_init_score:
        return best_r
    return x0

def _optimize_radii(init_radii, start_time, deadline):
    """Optimize radii using Nelder-Mead."""
    n = len(init_radii)
    
    def objective(radii):
        if time.time() > deadline - 5.0:
            return 1e9
        score = _score_from_radii(radii)
        if score < 0:
            return 1e9
        return -score
    
    x0 = init_radii
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    except Exception:
        return init_radii

    if time.time() > deadline - 5.0:
        return init_radii
    
    best_r = res.x
    best_score = -res.fun
    init_score = -objective(x0)
    
    if best_score > init_score:
        return best_r
    return x0

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses radial optimization to maximize the minimum vertex score.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial candidates for radii
    candidates = []
    
    # 1. Regular polygon
    candidates.append(np.ones(n))
    
    # 2. Symmetric wavy patterns
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.05, 0.1, 0.2]:
            radii = 1.0 + amp * np.cos(freq * angles)
            # Ensure positive
            radii = np.maximum(radii, 0.5)
            candidates.append(radii)
    
    best_polygon = None
    best_score = -1.0
    
    # Evaluate all initial candidates
    for radii in candidates:
        if time.time() > deadline - 60.0:
            break
        score = _score_from_radii(radii)
        if score > best_score:
            best_score = score
            # Reconstruct polygon
            p = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
            best_polygon = _normalize(p)
            
            # If we have a very good score, we might stop early?
            # But optimization might improve it further.
    
    if best_polygon is None:
        # Fallback to regular
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(p)
        best_score = _polygon_score(best_polygon)
        best_radii = np.ones(n)
    else:
        # Find the radii that gave best_score to use as start for optimization
        # Re-derive radii from best_polygon? No, we stored them implicitly.
        # Let's re-run the logic to find the best initial radii
        best_initial_radii = np.ones(n)
        max_init_score = -1.0
        for radii in candidates:
            s = _score_from_radii(radii)
            if s > max_init_score:
                max_init_score = s
                best_initial_radii = radii
        
        # Optimize from best initial radii using Smooth L-BFGS-B first
        if time.time() < deadline - 10.0:
            optimized_radii = _optimize_radii_smooth(best_initial_radii, start_time, deadline)
            
            # Check if optimized is better
            opt_score = _score_from_radii(optimized_radii)
            if opt_score > best_score:
                best_score = opt_score
                p = np.stack([optimized_radii * np.cos(angles), optimized_radii * np.sin(angles)], axis=1)
                best_polygon = _normalize(p)
                best_initial_radii = optimized_radii
        
        # Then try Nelder-Mead on the best so far
        if time.time() < deadline - 30.0:
            optimized_radii = _optimize_radii(best_initial_radii, start_time, deadline)
            opt_score = _score_from_radii(optimized_radii)
            if opt_score > best_score:
                best_score = opt_score
                p = np.stack([optimized_radii * np.cos(angles), optimized_radii * np.sin(angles)], axis=1)
                best_polygon = _normalize(p)

        # Try a few more random symmetric perturbations around best
        if time.time() < deadline - 30.0:
            rng = np.random.default_rng(42)
            for _ in range(10):
                # Add small noise
                noise = rng.normal(0, 0.05, n)
                # Keep symmetry? No, just random.
                test_radii = best_initial_radii + noise
                test_radii = np.maximum(test_radii, 0.5)
                
                if time.time() > deadline - 10.0:
                    break
                
                s = _score_from_radii(test_radii)
                if s > best_score:
                    # Optimize this one too with smooth method
                    if time.time() < deadline - 10.0:
                        opt_r = _optimize_radii_smooth(test_radii, start_time, deadline)
                        s_opt = _score_from_radii(opt_r)
                        if s_opt > best_score:
                            best_score = s_opt
                            p = np.stack([opt_r * np.cos(angles), opt_r * np.sin(angles)], axis=1)
                            best_polygon = _normalize(p)

    # Final check
    if not _is_valid_convex(best_polygon):
        # Fallback
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(p)

    return best_polygon

# EVOLVE-BLOCK-END
