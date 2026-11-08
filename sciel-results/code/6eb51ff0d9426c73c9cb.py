import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 14

def compute_min_triangle_area(pts):
    """Compute minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * np.abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
                if area < min_area:
                    min_area = area
    return min_area

def convex_hull_area(pts):
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except:
        # Fallback: use polygon area if all points are in convex position
        # Sort by angle from centroid
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
        order = np.argsort(angles)
        sorted_pts = pts[order]
        # Shoelace formula
        x = sorted_pts[:, 0]
        y = sorted_pts[:, 1]
        return 0.5 * np.abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))

def score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = compute_min_triangle_area(points)
    hull_area = convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def two_ring_points(n_inner, n_outer, r_inner, r_outer, angle_offset_inner, angle_offset_outer):
    """Generate points on two concentric circles."""
    pts = []
    # Inner ring
    for i in range(n_inner):
        angle = angle_offset_inner + 2 * np.pi * i / n_inner
        pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    # Outer ring
    for i in range(n_outer):
        angle = angle_offset_outer + 2 * np.pi * i / n_outer
        pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    return np.array(pts)

def log_sum_exp_min_triangle(pts, beta):
    """Compute log-sum-exp approximation of the minimum triangle area.
    
    We want to maximize min_area. Using log-sum-exp on negative areas:
    min(a) ≈ -(1/beta) * log(sum(exp(-beta * a)))
    
    So we minimize this smooth approximation.
    """
    n = len(pts)
    # Collect all triangle areas
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * np.abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
                areas.append(area)
    areas = np.array(areas)
    
    # Shift for numerical stability
    max_neg = -np.min(areas)  # = -min(areas)
    # log_sum_exp(-beta * areas) = max_neg * beta + log(sum(exp(-beta * areas - max_neg * beta)))
    shifted = -beta * areas - max_neg * beta
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    
    # min_area ≈ -(1/beta) * log_sum_exp(-beta * areas)
    # = -(1/beta) * (max_neg * beta + log_sum)
    # = -max_neg - (1/beta) * log_sum
    min_approx = -max_neg - (1.0 / beta) * log_sum
    return min_approx

def neg_score_two_ring(params, n_inner, n_outer, beta):
    """Objective for L-BFGS-B: negative of min triangle area (to minimize).
    
    params: [r_inner, r_outer, angle_offset_inner, angle_offset_outer]
    """
    r_inner, r_outer, a_off_in, a_off_out = params
    
    # Ensure positive radii
    r_inner = max(r_inner, 0.01)
    r_outer = max(r_outer, r_inner + 0.01)
    
    pts = two_ring_points(n_inner, n_outer, r_inner, r_outer, a_off_in, a_off_out)
    
    # Compute log-sum-exp approximation of min area
    min_area_approx = log_sum_exp_min_triangle(pts, beta)
    
    # We want to maximize min_area, so minimize -min_area
    return -min_area_approx

def optimize_two_ring(n_inner, n_outer, init_params, beta, deadline):
    """Optimize two-ring configuration using L-BFGS-B."""
    start_time = time.time()
    
    # Initial bounds
    bounds = [(0.05, 1.0), (0.1, 2.0), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    best_x = np.array(init_params)
    best_f = neg_score_two_ring(best_x, n_inner, n_outer, beta)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            neg_score_two_ring,
            best_x,
            args=(n_inner, n_outer, beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000}
        )
        if result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_x
    
    # Final polish with higher beta
    beta_high = beta * 5
    try:
        result2 = minimize(
            neg_score_two_ring,
            best_x,
            args=(n_inner, n_outer, beta_high),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000}
        )
        if result2.fun < best_f:
            best_x = result2.x
            best_f = result2.fun
    except:
        pass
    
    return best_x

def construct():
    deadline = 120.0  # Safety margin
    start_time = time.time()
    
    best_score = 0.0
    best_points = None
    
    # Try different splits of 14 points between two rings
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2), (1, 13), (13, 1), (14, 0), (0, 14)]
    
    beta = 50.0  # Temperature for log-sum-exp
    
    for n_inner, n_outer in splits:
        if n_inner == 0 or n_outer == 0:
            continue  # Skip degenerate cases for now
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Multiple random initializations
        n_init = 3
        for init_idx in range(n_init):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            # Random initial parameters
            r_inner_init = 0.2 + 0.3 * (init_idx / max(n_init-1, 1))
            r_outer_init = 0.6 + 0.3 * (init_idx / max(n_init-1, 1))
            a_off_in_init = np.random.uniform(-np.pi/4, np.pi/4)
            a_off_out_init = np.random.uniform(-np.pi/4, np.pi/4)
            
            init_params = [r_inner_init, r_outer_init, a_off_in_init, a_off_out_init]
            
            result_params = optimize_two_ring(n_inner, n_outer, init_params, beta, deadline - 5)
            
            # Evaluate actual score
            pts = two_ring_points(n_inner, n_outer, result_params[0], result_params[1], 
                                  result_params[2], result_params[3])
            s = score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    
    # If no good result found, try a simple regular polygon as fallback
    if best_points is None or best_score < 1e-6:
        # Regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = score(best_points)
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > deadline - 2:
        # Return best found so far
        pass
    
    return best_points
