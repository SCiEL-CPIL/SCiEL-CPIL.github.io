# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-END


def _normalize(polygon):
    """Subtract centroid, divide by mean distance to centroid."""
    c = polygon.mean(axis=0)
    p = polygon - c
    dists = np.linalg.norm(p, axis=1)
    mean_d = dists.mean()
    if mean_d <= 0.01:
        return None
    return p / mean_d


def _is_strictly_convex(p):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turn +-2pi."""
    n = len(p)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cp = e1[0] * e2[1] - e1[1] * e2[0]
        cross_products.append(cp)
    cross_products = np.array(cross_products)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-10):
        return False
    signs = np.sign(cross_products)
    if len(set(signs)) != 1:
        return False
    # Total turning should be +-2*pi
    angles = np.arctan2(cross_products, 
                        np.array([(p[(i+1)%n] - p[i]) @ (p[(i+2)%n] - p[(i+1)%n]) for i in range(n)]))
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return False
    return True


def _vertex_score(v, polygon):
    """Compute the score for a single vertex v in the polygon."""
    dists = np.linalg.norm(polygon - v, axis=1)
    other_dists = np.sort(dists[dists > 1e-10])
    
    if len(other_dists) < 19:
        return 0.0
    
    d = other_dists[:19]
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    D = (d[k] + d[k + 1]) / 2
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the overall score: minimum vertex score over all vertices."""
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(polygon[i], polygon)
    return np.min(scores)


def _evaluate(polygon):
    """Full evaluation: normalize, check convexity, compute score."""
    normed = _normalize(polygon)
    if normed is None:
        return 0.0
    if not _is_strictly_convex(normed):
        return 0.0
    return _polygon_score(normed)


def _regular_polygon(n=20, radius=1.0):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _smooth_min(scores, tau=0.01):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -log(sum(exp(-x/tau))) * tau
    This is differentiable and approximates min as tau -> 0.
    """
    scores = np.asarray(scores)
    # Shift for numerical stability
    shifted = -scores / tau
    m = np.max(shifted)
    if m > 700: # overflow protection
        return -np.min(scores)
    s = np.sum(np.exp(shifted - m))
    return -(m + np.log(s)) * tau


def _objective_direct(coeffs, tau=0.01):
    """
    Objective function for direct vertex optimization.
    coeffs: 40-dimensional vector of vertex coordinates.
    Returns negative of smooth minimum vertex score + convexity penalty.
    """
    n = 20
    poly = coeffs.reshape(n, 2)
    
    normed = _normalize(poly)
    if normed is None:
        return 1000.0
    
    # Convexity check and penalty
    # We want all cross products to be positive (CCW) and non-zero
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = normed[(i + 1) % n] - normed[i]
        e2 = normed[(i + 2) % n] - normed[(i + 1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    
    # Penalty for non-convexity
    # If any cross product is <= 0, it's not strictly convex CCW.
    # We use a soft penalty: sum of max(0, -cp)^2
    penalty = np.sum(np.maximum(0, -cross_products)**2)
    
    # Also check for repeated points (should be rare in continuous search)
    # But the cross product penalty handles most degeneracies.
    
    # Compute vertex scores
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(normed[i], normed)
    
    # Smooth minimum
    sm = _smooth_min(scores, tau)
    
    # Total objective: minimize -sm + penalty
    # We want to maximize sm, so we minimize -sm.
    # The penalty discourages non-convex shapes.
    return -sm + penalty


def _search_direct(time_budget=600, deadline_margin=30):
    """
    Search for best polygon using direct vertex coordinate optimization.
    """
    start_time = time.time()
    deadline = start_time + time_budget - deadline_margin
    
    # Initial guess: regular polygon
    best_poly = _regular_polygon(20)
    best_score = _evaluate(best_poly)
    best_coeffs = best_poly.flatten()
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial coefficients
        if restart == 0:
            init = best_coeffs.copy()
        else:
            # Jitter the regular polygon or perturb the best
            if restart % 2 == 0:
                # Jitter regular
                base = _regular_polygon(20)
                noise = np.random.normal(0, 0.1, (20, 2))
                init = (base + noise).flatten()
            else:
                # Perturb best
                init = best_coeffs + np.random.normal(0, 0.05, 40)
        
        # Evaluate initial
        init_poly = init.reshape(20, 2)
        init_score = _evaluate(init_poly)
        if init_score > best_score:
            best_score = init_score
            best_poly = init_poly.copy()
            best_coeffs = init.copy()
        
        # Run optimization
        try:
            # Use L-BFGS-B for efficiency
            # We don't have bounds, but L-BFGS-B can handle unbounded
            # However, we might want to keep coordinates in a reasonable range
            # The normalization step handles scaling, so absolute magnitude doesn't matter much
            # but extreme values can cause numerical issues.
            
            result = minimize(
                _objective_direct,
                init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Evaluate the result
            result_poly = result.x.reshape(20, 2)
            result_score = _evaluate(result_poly)
            
            if result_score > best_score:
                best_score = result_score
                best_poly = result_poly.copy()
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final refinement with tighter tolerance and smaller tau
    if time.time() < deadline - 10:
        try:
            def _objective_refined(coeffs):
                return _objective_direct(coeffs, tau=0.001)
            
            result = minimize(
                _objective_refined,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            result_poly = result.x.reshape(20, 2)
            result_score = _evaluate(result_poly)
            
            if result_score > best_score:
                best_score = result_score
                best_poly = result_poly.copy()
                
        except Exception:
            pass
    
    return best_poly, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Uses direct vertex coordinate optimization with a smooth surrogate for the min score.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    result, score = _search_direct(time_budget=600, deadline_margin=30)
    
    # Validate the result
    final_score = _evaluate(result)
    if final_score <= 0:
        # Fallback to regular polygon
        return _regular_polygon(20)
    
    return result


# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
