# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize
import random

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _get_triangle_areas(points):
    """
    Efficiently compute all triangle areas and return the array.
    """
    n = points.shape[0]
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _smoothed_objective(params, logsumexp_beta=50.0):
    """
    Smooth surrogate for the min-triangle-area objective using log-sum-exp.
    This allows gradient-based optimization (L-BFGS-B) to coordinate all 26 coordinates.
    """
    n = 13
    points = params.reshape(n, 2)
    
    # Compute all triangle areas
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Use log-sum-exp for smooth min approximation
    # soft_min = -logsumexp(-beta * areas) / beta
    # For large beta, this approaches min(areas)
    # But we want to maximize soft_min, so minimize -soft_min
    # -soft_min = logsumexp(-beta * areas) / beta
    
    # Actually, let's just use a simple soft-min via log-sum-exp
    # soft_min(x) = - (1/beta) * log(sum(exp(-beta * x_i)))
    # We want to maximize soft_min, so minimize -soft_min
    # -soft_min = (1/beta) * log(sum(exp(-beta * x_i)))
    
    # To avoid numerical issues, use the stable log-sum-exp
    max_neg_beta_areas = np.max(-logsumexp_beta * areas)
    log_sum_exp = max_neg_beta_areas + np.log(np.sum(np.exp(-logsumexp_beta * areas - max_neg_beta_areas)))
    soft_min = -log_sum_exp / logsumexp_beta
    
    # We want to maximize soft_min / A_hull
    # But A_hull is hard to compute smoothly. Let's just maximize soft_min for now,
    # or use a fixed normalization.
    # Actually, the score is A_min / A_hull.
    # If we fix the points to be in a unit square [0,1]^2, A_hull is bounded.
    # Let's just maximize soft_min directly, as A_hull is roughly constant for convex configs.
    # Or better: compute the actual score for the current params, but use the gradient of the smoothed objective.
    
    # For optimization, we minimize -soft_min.
    return -soft_min

def _smoothed_objective_grad(params, logsumexp_beta=50.0):
    """
    Gradient of the smoothed objective.
    """
    n = 13
    points = params.reshape(n, 2)
    
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min gradient
    # f = - (1/beta) * log(sum(exp(-beta * areas)))
    # df/darea_i = (1/beta) * (exp(-beta * area_i) / sum(exp(-beta * area_j)))
    #             = (1/beta) * softmax(-beta * areas)[i]
    
    max_neg_beta_areas = np.max(-logsumexp_beta * areas)
    exp_terms = np.exp(-logsumexp_beta * areas - max_neg_beta_areas)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # These are the softmax weights
    
    # Gradient of soft_min w.r.t areas:
    # d(soft_min)/d(area_i) = weights[i]
    
    # Now, gradient of area_i w.r.t points.
    # area = 0.5 * |cross|
    # d(area)/d(cross) = 0.5 * sign(cross)
    # cross = v1_x * v2_y - v1_y * v2_x
    # v1 = p1 - p0, v2 = p2 - p0
    # cross = (p1_x - p0_x)(p2_y - p0_y) - (p1_y - p0_y)(p2_x - p0_x)
    
    # Let's compute the gradient of the cross product for each triangle, then weight by sign and areas gradient.
    # This is getting complex. Let's just use finite differences or a simpler gradient.
    # Given time constraints, let's just use L-BFGS-B with numerical gradients or a simple heuristic.
    # Actually, let's just use the fact that we can compute the exact gradient of the min-area if we know which triangle is min.
    # But we're smoothing it.
    
    # Let's just return a zero gradient for now and rely on the optimizer's numerical gradient? No, that's slow.
    # Let's implement the gradient properly.
    
    # Gradient of area_i w.r.t p0, p1, p2:
    # area = 0.5 * | (p1-p0) x (p2-p0) |
    # Let C = (p1-p0) x (p2-p0)
    # d(area)/d(p0) = 0.5 * sign(C) * ( - (p2-p0) + (p1-p0) ) ? No.
    # C = (p1_x - p0_x)(p2_y - p0_y) - (p1_y - p0_y)(p2_x - p0_x)
    # dC/dp0_x = -(p2_y - p0_y) + (p1_y - p0_y) = p1_y - p2_y
    # dC/dp0_y = -(p2_x - p0_x) + (p1_x - p0_x) = p1_x - p2_x
    # dC/dp1_x = (p2_y - p0_y)
    # dC/dp1_y = -(p2_x - p0_x)
    # dC/dp2_x = -(p1_y - p0_y)
    # dC/dp2_y = (p1_x - p0_x)
    
    # So d(area)/dp0 = 0.5 * sign(C) * (p1_y - p2_y, p1_x - p2_x)
    # d(area)/dp1 = 0.5 * sign(C) * (p2_y - p0_y, -(p2_x - p0_x))
    # d(area)/dp2 = 0.5 * sign(C) * (-(p1_y - p0_y), p1_x - p0_x)
    
    # Total gradient of soft_min w.r.t point k is sum over all triangles containing k of weights[i] * d(area_i)/d(point_k)
    
    grad = np.zeros_like(points)
    
    for i in range(len(triples)):
        w = weights[i]
        if w < 1e-8:
            continue
            
        i0, i1, i2 = triples[i]
        c = cross[i]
        s = np.sign(c)
        if s == 0:
            s = 1.0
            
        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        
        # d(area)/dp0
        grad[i0] += w * 0.5 * s * np.array([p1[1] - p2[1], p1[0] - p2[0]])
        
        # d(area)/dp1
        grad[i1] += w * 0.5 * s * np.array([p2[1] - p0[1], -(p2[0] - p0[0])])
        
        # d(area)/dp2
        grad[i2] += w * 0.5 * s * np.array([-(p1[1] - p0[1]), p1[0] - p0[0]])
        
    # We are minimizing -soft_min, so the gradient of the objective is -grad
    return -grad

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Smoothed Objective Gradient Optimization.
    """
    start_time = time.time()
    deadline = 630  # seconds, strict limit
    safety_margin = 10
    effective_deadline = deadline - safety_margin
    
    # Initial configurations
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r = 0.4
    pts_reg = np.column_stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)])
    
    # 2. Hexagonal lattice subset
    lattice_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    sorted_indices = np.argsort(dists)
    hex_pts = lattice_pts[sorted_indices[:13]]
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    max_norm = np.max(np.linalg.norm(hex_pts, axis=1))
    if max_norm > 0:
        hex_pts = hex_pts / max_norm * 0.4 + 0.5
        
    # 3. Random
    rng = np.random.default_rng(42)
    pts_rand = rng.uniform(0.1, 0.9, (13, 2))
    
    initializations = [pts_reg, hex_pts, pts_rand]
    
    best_points = None
    best_score = -np.inf
    
    for init_pts in initializations:
        if time.time() - start_time > effective_deadline:
            break
            
        # Flatten to 1D for optimization
        x0 = init_pts.flatten()
        
        # Run L-BFGS-B
        # We minimize -soft_min
        # Note: L-BFGS-B might be slow with 26 variables and 286 triangles.
        # Let's limit iterations.
        
        try:
            result = minimize(
                _smoothed_objective,
                x0,
                jac=_smoothed_objective_grad,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * 26,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50}
            )
            
            opt_pts = result.x.reshape(13, 2)
            opt_score = _compute_score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception as e:
            pass
            
    # If no good result, use the best initial
    if best_points is None:
        best_points = initializations[0]
        best_score = _compute_score(best_points)
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
