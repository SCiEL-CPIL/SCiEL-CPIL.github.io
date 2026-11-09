# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
import sys

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


def _objective_direct(x, start_time, deadline):
    """
    Objective function for differential evolution.
    x is a flat array of 40 values (20 x-coords, 20 y-coords).
    Returns a scalar to be minimized.
    """
    if time.time() > deadline:
        return 1e10
    
    poly = x.reshape(20, 2)
    normed = _normalize(poly)
    if normed is None:
        return 1e10
    
    # Check convexity
    if not _is_strictly_convex(normed):
        # Penalty for non-convexity: large value
        # We can compute a penalty based on cross products
        n = 20
        cross_products = []
        for i in range(n):
            e1 = normed[(i + 1) % n] - normed[i]
            e2 = normed[(i + 2) % n] - normed[(i + 1) % n]
            cp = e1[0] * e2[1] - e1[1] * e2[0]
            cross_products.append(cp)
        cross_products = np.array(cross_products)
        # If any cross product is negative (assuming CCW), it's not convex
        # We want all cross products to have the same sign.
        # Let's assume we want CCW (positive cross products).
        # Penalty = sum of negative cross products squared + sum of small positive cross products?
        # Simpler: if not strictly convex, return a penalty proportional to violation.
        min_cp = np.min(cross_products)
        if min_cp < 0:
            return 1e6 + abs(min_cp)
        if np.any(np.abs(cross_products) < 1e-5):
            return 1e6
        # Check total turn
        angles = np.arctan2(cross_products, 
                            np.array([(normed[(i+1)%n] - normed[i]) @ (normed[(i+2)%n] - normed[(i+1)%n]) for i in range(n)]))
        total_turn = np.sum(angles)
        if abs(abs(total_turn) - 2 * np.pi) > 0.1:
            return 1e5 + abs(abs(total_turn) - 2 * np.pi)
        
        return 1e4 # Valid convexity but maybe not strictly enough? No, _is_strictly_convex handles that.
        # Actually, if _is_strictly_convex returns False, we should penalize.
        # The above checks are a bit redundant. Let's just return a large penalty if not strictly convex.
        # But we need a gradient-like signal.
        # Let's use the sum of squares of negative cross products (assuming CCW).
        neg_cps = cross_products[cross_products < 0]
        penalty = np.sum(neg_cps**2) if len(neg_cps) > 0 else 0
        # Also penalize if total turn is wrong
        turn_err = abs(abs(total_turn) - 2 * np.pi)
        return 1e3 + penalty + turn_err

    # Compute vertex scores
    scores = np.array([_vertex_score(normed[i], normed) for i in range(20)])
    min_score = np.min(scores)
    
    # We want to maximize min_score, so minimize -min_score
    # Add a small penalty for variance to encourage uniformity?
    # The task metric is the minimum. So maximizing the minimum is the goal.
    return -min_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Uses differential evolution over the 40-dimensional vertex space.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600 - 30 # 30s safety margin
    
    # Initial population: Regular polygon
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    init_poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    init_x = init_poly.flatten()
    
    # Define bounds for differential evolution
    # The polygon will be normalized, so the absolute scale doesn't matter much,
    # but we need to keep it within a reasonable range to avoid numerical issues.
    # A regular polygon has radius 1. Let's allow radius to vary between 0.5 and 2.
    bounds = [(-2.0, 2.0)] * 40
    
    # Differential Evolution
    # maxiter: max number of iterations
    # popsize: population size (multiplier)
    # tol: tolerance for stopping
    # strategy: 'best1bin' is standard
    # mutation: (0.5, 1.0)
    # recombination: 0.7
    
    try:
        result = differential_evolution(
            _objective_direct,
            bounds,
            args=(start_time, deadline),
            maxiter=50,
            popsize=20,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True, # Use L-BFGS-B for final polish
            init='latinhypercube'
        )
        
        best_x = result.x
        best_score = -result.fun # Since we minimized -score
        
    except Exception as e:
        print(f"Differential Evolution failed: {e}")
        best_x = init_x
        best_score = _evaluate(init_poly)
    
    # Check time
    if time.time() > deadline:
        poly = best_x.reshape(20, 2)
        return poly
    
    # Final local refinement with L-BFGS-B on the exact objective
    # We need a wrapper for L-BFGS-B that takes a flat array and returns a scalar
    def _obj_l_bfgs(x):
        poly = x.reshape(20, 2)
        normed = _normalize(poly)
        if normed is None:
            return 1e10
        if not _is_strictly_convex(normed):
            return 1e10
        scores = np.array([_vertex_score(normed[i], normed) for i in range(20)])
        return -np.min(scores)
    
    from scipy.optimize import minimize
    try:
        res_l_bfgs = minimize(
            _obj_l_bfgs,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res_l_bfgs.fun < -best_score: # Better score (lower objective)
            best_x = res_l_bfgs.x
            best_score = -res_l_bfgs.fun
    except Exception:
        pass
    
    final_poly = best_x.reshape(20, 2)
    
    # Validate
    final_score = _evaluate(final_poly)
    if final_score <= 0:
        return init_poly
    
    return final_poly


# EVOLVE-BLOCK-START
import numpy as np
import time
# EVOLVE-BLOCK-END
