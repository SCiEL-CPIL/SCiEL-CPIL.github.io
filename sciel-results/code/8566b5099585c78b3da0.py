"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_points(r_inner, theta_offset):
    """
    Generate 14 points: 7 on unit circle, 7 on circle of radius r_inner.
    The inner points are rotated by theta_offset relative to the outer points.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _negative_score(params):
    """
    Objective function for optimization.
    params: [r_inner, theta_offset]
    Returns negative score to allow minimization.
    """
    r_inner, theta_offset = params
    
    # Constraints: r_inner must be positive and less than 1
    # theta_offset can be any real, but due to symmetry, [0, 2*pi/7) is sufficient
    if r_inner < 0.01 or r_inner > 0.99:
        return 1e9 # Penalize invalid radii
    
    pts = _generate_points(r_inner, theta_offset)
    score = _score(pts)
    
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    best_score = -float('inf')
    best_params = None
    
    # Strategy: Exploit symmetry (2 concentric heptagons)
    # 1. Coarse Grid Search
    # r_inner in [0.1, 0.9], theta_offset in [0, 2*pi/7)
    # The symmetry of the heptagon means theta_offset has period 2*pi/7
    
    r_range = np.linspace(0.1, 0.9, 50)
    theta_range = np.linspace(0, 2*np.pi/7, 50, endpoint=False)
    
    coarse_best_score = -float('inf')
    coarse_best_r = 0.5
    coarse_best_theta = 0.0
    
    for r in r_range:
        for theta in theta_range:
            if time.time() > deadline:
                break
            pts = _generate_points(r, theta)
            s = _score(pts)
            if s > coarse_best_score:
                coarse_best_score = s
                coarse_best_r = r
                coarse_best_theta = theta
        
        if time.time() > deadline:
            break
    
    best_score = coarse_best_score
    best_params = [coarse_best_r, coarse_best_theta]
    
    # 2. Local Refinement using Nelder-Mead
    # Start from the best point found in the grid search
    # We might also try a few other local optima if time permits
    
    if time.time() < deadline - 10.0:
        # Define bounds for Nelder-Mead (it doesn't natively support bounds well, 
        # but we can penalize in the objective or use 'L-BFGS-B' if we had gradients.
        # Since _score is non-differentiable (min of areas), Nelder-Mead is a good choice.
        # We will run it starting from the coarse best.
        
        x0 = np.array([coarse_best_r, coarse_best_theta])
        
        try:
            res = minimize(_negative_score, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
            
            if res.success or (not res.success and res.fun < -coarse_best_score):
                r_opt, theta_opt = res.x
                # Ensure parameters are within reasonable bounds
                if 0.01 < r_opt < 0.99:
                    pts_opt = _generate_points(r_opt, theta_opt)
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_params = [r_opt, theta_opt]
        except Exception:
            pass
            
    # 3. Try a few other random starts for local optimization if time remains
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            r_rand = rng.uniform(0.1, 0.9)
            theta_rand = rng.uniform(0, 2*np.pi/7)
            
            try:
                res = minimize(_negative_score, np.array([r_rand, theta_rand]), method='Nelder-Mead', 
                               options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 500})
                
                if res.success or (not res.success and res.fun < -best_score):
                    r_opt, theta_opt = res.x
                    if 0.01 < r_opt < 0.99:
                        pts_opt = _generate_points(r_opt, theta_opt)
                        s_opt = _score(pts_opt)
                        if s_opt > best_score:
                            best_score = s_opt
                            best_params = [r_opt, theta_opt]
            except Exception:
                pass

    # Final construction
    if best_params is not None:
        final_pts = _generate_points(best_params[0], best_params[1])
    else:
        # Fallback to default if something went wrong
        final_pts = _generate_points(0.5, 0.0)
        
    return final_pts
# EVOLVE-BLOCK-END
