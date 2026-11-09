# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

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
    # Compute angles of edges
    angles = np.arctan2(cross_products, 
                        np.array([(p[(i+1)%n] - p[i]) @ (p[(i+2)%n] - p[(i+1)%n]) for i in range(n)]))
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return False
    return True


def _vertex_score(v, polygon):
    """Compute the score for a single vertex v in the polygon."""
    # Compute distances from v to all other vertices
    dists = np.linalg.norm(polygon - v, axis=1)
    # Exclude the vertex itself (distance 0)
    other_dists = np.sort(dists[dists > 1e-10])
    
    if len(other_dists) < 19:
        return 0.0
    
    d = other_dists[:19]
    
    # If any distance < 0.05, score is 0
    if d[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (d[k] + d[k + 1]) / 2
    
    # If D < 0.1, score is 0
    if D < 0.1:
        return 0.0
    
    # Choose four distances
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    # raw = mean of max(d/D, D/d) over the four distances
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the overall score: minimum vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        s = _vertex_score(polygon[i], polygon)
        scores.append(s)
    return min(scores)


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


def _polygon_from_params(params):
    """
    Reconstruct polygon from 19 radial perturbations.
    params: array of 19 values, perturbations to radii of vertices 1..19.
    Vertex 0 radius is fixed at 1.0.
    """
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    radii[1:] = 1.0 + params
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))


def _smooth_min(scores, temperature=0.05):
    """
    Smooth approximation of the minimum of scores.
    Uses log-sum-exp trick for stability.
    min(scores) ≈ -1/T * log(sum(exp(-T * scores)))
    """
    if len(scores) == 0:
        return 0.0
    # For numerical stability, subtract max (which is -min of -scores)
    # We want min(scores), so we use -logsumexp(-scores)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # -logsumexp(-scores) = -max(-scores) - log(sum(exp(-scores - max(-scores))))
    # max(-scores) = -min(scores)
    # So -logsumexp(-scores) = min(scores) - log(sum(exp(-scores + min(scores))))
    # This is a smooth lower bound on min(scores)
    
    # Actually, let's use the standard smooth min:
    # m_T(s) = -1/T * log( sum_i exp(-T * s_i) )
    # As T -> infinity, this approaches min(s_i).
    
    # To avoid overflow, we can factor out the minimum:
    # sum exp(-T s_i) = exp(-T * min_s) * sum exp(-T (s_i - min_s))
    # log(sum) = -T * min_s + log(sum exp(-T (s_i - min_s)))
    # m_T = min_s - 1/T * log(sum exp(-T (s_i - min_s)))
    
    # Since s_i - min_s >= 0, exp(-T (s_i - min_s)) <= 1.
    # Sum is at least 1 (from the min term).
    # So log(sum) >= 0.
    # m_T <= min_s.
    
    # This is a good smooth approximation.
    
    s = np.array(scores)
    min_s = np.min(s)
    # If min_s is very negative, exp(-T*(s-min_s)) might be fine since s-min_s >= 0.
    # But if scores are large, T*score might overflow.
    # Let's just use the formula directly with clipping if necessary.
    
    # For our scores, they are in [0, 1]. So T*s is small.
    exp_terms = np.exp(-temperature * (s - min_s))
    sum_exp = np.sum(exp_terms)
    return min_s - (1.0 / temperature) * np.log(sum_exp)


def _objective(params):
    """
    Objective function for L-BFGS-B.
    We want to maximize the minimum vertex score.
    So we minimize the negative of a smooth approximation of the min score.
    """
    poly = _polygon_from_params(params)
    normed = _normalize(poly)
    if normed is None:
        return 100.0
    
    if not _is_strictly_convex(normed):
        # Penalize non-convexity
        # Compute cross products to measure violation
        n = len(normed)
        min_cross = float('inf')
        for i in range(n):
            e1 = normed[(i + 1) % n] - normed[i]
            e2 = normed[(i + 2) % n] - normed[(i + 1) % n]
            cp = e1[0] * e2[1] - e1[1] * e2[0]
            if cp < min_cross:
                min_cross = cp
        # If any cross product is negative or zero, it's not strictly convex (assuming CCW)
        # We assume the regular polygon is CCW.
        # If min_cross <= 0, penalty.
        if min_cross <= 1e-6:
            return 100.0 - min_cross * 10
        
    scores = []
    for i in range(len(normed)):
        scores.append(_vertex_score(normed[i], normed))
    
    min_score = _smooth_min(scores, temperature=0.1)
    return -min_score


def _search_optimized():
    """
    Multi-start L-BFGS-B optimization in the 19-dimensional parameter space.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -1.0
    best_params = np.zeros(19)
    
    # Initial points
    # 1. Regular polygon (all zeros)
    # 2. Symmetric alternating (from previous best strategy, but mapped to params)
    #    Symmetric: radii alternate 1, r, 1, r...
    #    params[0] = r-1, params[1] = 0, params[2] = r-1...
    #    Let's try a few ratios
    r_ratios = [0.8, 0.9, 0.95, 0.7, 0.85]
    
    initial_points = [np.zeros(19)]
    for r in r_ratios:
        p = np.zeros(19)
        # Vertices 1, 3, 5... are odd indices in the polygon.
        # In params, index i corresponds to vertex i+1.
        # Vertex 1 (odd) -> params[0]
        # Vertex 3 (odd) -> params[2]
        # So even indices in params correspond to odd vertices in polygon.
        p[0::2] = r - 1.0
        initial_points.append(p)
        
    # 3. Random perturbations
    np.random.seed(42)
    for _ in range(5):
        p = np.random.normal(0, 0.1, 19)
        initial_points.append(p)
        
    # 4. More structured randoms
    for _ in range(5):
        p = np.random.uniform(-0.2, 0.2, 19)
        initial_points.append(p)

    for init_p in initial_points:
        if time.time() > deadline:
            break
            
        try:
            result = minimize(
                _objective, 
                init_p, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * 19, # Radii between 0.5 and 1.5
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < -best_score:
                # Evaluate the actual score of the result
                poly = _polygon_from_params(result.x)
                score = _evaluate(poly)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
                    
        except Exception as e:
            pass
            
    return best_params, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses multi-start L-BFGS-B optimization in a 19D radial parameter space.
    """
    start_time = time.time()
    
    # 1. Run the optimized search
    best_params, best_score = _search_optimized()
    
    # 2. Construct the polygon from the best parameters
    best_poly = _polygon_from_params(best_params)
    
    # 3. Fallback to regular polygon if something went wrong
    if best_score <= 0:
        return _regular_polygon(20)
        
    return best_poly


# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
