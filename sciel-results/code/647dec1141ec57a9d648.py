import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _triangle_areas(points):
    """
    Compute areas of all triangles formed by the points.
    Returns an array of shape (num_triangles,)
    """
    n = len(points)
    indices = np.arange(n)
    # Generate all combinations of 3 indices
    # For n=13, C(13,3) = 286, which is small
    tri_indices = list(combinations(range(n), 3))
    
    areas = np.empty(len(tri_indices))
    for i, (i1, i2, i3) in enumerate(tri_indices):
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas[i] = 0.5 * abs(cross)
    return areas

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the raw score: min triangle area / convex hull area.
    """
    if _hull_area(points) < 1e-10:
        return 0.0
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / _hull_area(points)

def _soft_min_score(points, beta):
    """
    Compute a smooth approximation of the score using log-sum-exp for the minimum.
    This is differentiable and suitable for L-BFGS-B.
    """
    areas = _triangle_areas(points)
    hull_area = _hull_area(points)
    
    if hull_area < 1e-10:
        return 0.0
    
    # To maximize min(areas) / hull_area, we can minimize -log(sum(exp(-beta * areas))) / hull_area
    # Or more simply, maximize a soft-min of the areas normalized by hull area.
    # Let's maximize: soft_min(areas) / hull_area
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    
    # For numerical stability with large beta:
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize this.
    
    # Let's use a simpler surrogate: 
    # Objective to minimize: - ( -1/beta * log(sum(exp(-beta * areas))) ) / hull_area
    # Which is: (1/beta * log(sum(exp(-beta * areas)))) / hull_area
    
    # However, areas are small. Let's normalize areas by hull_area first?
    # No, the score is min_area / hull_area.
    # Let's define f(points) = min_area / hull_area.
    # Approximation: 
    # S = sum(exp(-beta * areas))
    # log_S = log(S)
    # soft_min = -1/beta * log_S
    # score_approx = soft_min / hull_area
    
    # We want to maximize score_approx, so minimize -score_approx.
    
    # To avoid overflow/underflow in exp(-beta * areas):
    # areas are typically ~0.01 to 0.1. beta ~ 100-1000.
    # beta * areas ~ 1 to 100. exp(-100) is small. exp(-1) is ~0.3.
    # This should be stable.
    
    beta_areas = -beta * areas
    # Subtract max for stability
    max_ba = np.max(beta_areas)
    if not np.isfinite(max_ba):
        return 0.0
    shifted = beta_areas - max_ba
    sum_exp = np.sum(np.exp(shifted))
    
    if sum_exp == 0:
        return 0.0
        
    log_sum = np.log(sum_exp) + max_ba
    soft_min = -1.0 / beta * log_sum
    
    return soft_min / hull_area

def _neg_soft_min_score(params, beta):
    """
    Objective function for L-BFGS-B.
    params: array of shape (13,) representing radial offsets.
    """
    # Reconstruct points from radial offsets
    # Base points: Regular 13-gon on unit circle (or centered at 0)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_r = 1.0
    base_x = base_r * np.cos(theta)
    base_y = base_r * np.sin(theta)
    
    # Apply radial offsets
    # Ensure points remain in a convex position.
    # If offsets are too large, convexity might break, but for small offsets it's fine.
    # We constrain offsets to be within [-0.5, 0.5] to ensure no point crosses the center or becomes non-convex relative to neighbors.
    r = base_r + params
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y))
    
    # Calculate score
    score = _soft_min_score(points, beta)
    
    # We want to maximize score, so minimize negative score
    return -score

def _regular_polygon():
    """
    Generate a regular 13-gon centered at origin.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _center_ring(radius, n_pts, phase=0.0):
    """
    Generate points on a circle of given radius.
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + phase
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = _regular_polygon()
    
    # Define the parameter space for radial offsets
    # We start with 0 offsets (regular polygon)
    # Bounds: [-0.5, 0.5] to keep points roughly in the unit disk and maintain convexity
    
    bounds = [(-0.5, 0.5)] * N
    
    # Strategy: Multi-start L-BFGS-B with different beta values (annealing)
    # Start with low beta (smooth) and increase to high beta (sharp min)
    
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Initial guesses for parameters (radial offsets)
    # 1. Zero (Regular Polygon)
    # 2. Small random noise
    # 3. Alternating pattern (to break symmetry)
    
    inits = [
        np.zeros(N),
        np.random.normal(0, 0.1, N),
        np.array([0.1 if i % 2 == 0 else -0.1 for i in range(N)]),
        np.random.uniform(-0.2, 0.2, N)
    ]
    
    for beta in betas:
        for init_params in inits:
            if time.time() > deadline - 10.0:
                break
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _neg_soft_min_score,
                    init_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 0: # res.fun is -score, so lower is better
                    optimal_params = res.x
                    # Reconstruct points
                    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
                    r = 1.0 + optimal_params
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    candidate_points = np.column_stack((x, y))
                    
                    # Evaluate true score (not soft min)
                    true_score = _score(candidate_points)
                    
                    if true_score > best_score:
                        best_score = true_score
                        best_points = candidate_points
                        # Update init for next beta to be the current best
                        inits[0] = optimal_params.copy()
                        
            except Exception:
                pass
        
        # After each beta stage, ensure inits[0] is the best found so far for continuity
        # If we found a better point, update the "zero" init to be that point's offsets
        # This helps the next, sharper beta start from a good place.
        
    # Final check: if we haven't improved, return the regular polygon
    # But usually, the optimization will improve upon the regular polygon.
    
    return best_points
