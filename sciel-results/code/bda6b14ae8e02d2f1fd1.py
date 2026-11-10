"""Program to maximize the product of distances for 8 planar points with fixed diameter."""
import numpy as np
import time
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    
    # Scale to diameter 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    mask = ~np.eye(n, dtype=bool)
    # Avoid log(0) if points coincide
    if np.any(dists_scaled[mask] < 1e-12):
        return 0.0
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score_from_params(params):
    """
    Compute negative log score for a C4 symmetric configuration defined by:
    params = [r1, theta2, r2]
    Orbit 1: r1 * exp(i * (0, pi/2, pi, 3pi/2))
    Orbit 2: r2 * exp(i * (theta2, theta2 + pi/2, ...))
    """
    r1 = params[0]
    theta2 = params[1]
    r2 = params[2]
    
    # Generate points
    pts1 = np.array([
        [r1 * cos(0), r1 * sin(0)],
        [r1 * cos(pi/2), r1 * sin(pi/2)],
        [r1 * cos(pi), r1 * sin(pi)],
        [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
    ])
    
    pts2 = np.array([
        [r2 * cos(theta2), r2 * sin(theta2)],
        [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
        [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
        [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
    ])
    
    pts = np.vstack([pts1, pts2])
    
    # Calculate score
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
        
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    
    # Check for zero distances
    if np.any(dists_scaled[mask] < 1e-12):
        return 1e10
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    neg_log_score = n * log(n) - log_prod
    return neg_log_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    best_score = 0.0
    best_pts = None
    
    # 1. Try C4 symmetric optimization (3 variables)
    # Initial guesses: Regular octagon has r1=1, r2=1, theta2=pi/4.
    # Since we scale to diameter 2, the absolute scale doesn't matter for the shape,
    # but it matters for the intermediate values.
    # Regular octagon diameter is 2R. If R=1, diam=2.
    # So r1=1, r2=1, theta2=pi/4 is a good start.
    
    try:
        # Use Differential Evolution for the 3D space to find global optimum in C4 manifold
        # Bounds: r in [0.1, 2.0], theta in [0, pi/2]
        bounds_c4 = [(0.1, 2.0), (0.0, pi/2), (0.1, 2.0)]
        de_res_c4 = differential_evolution(_neg_log_score_from_params, bounds_c4, maxiter=50, popsize=15, seed=42, tol=1e-6, polish=False)
        
        if de_res_c4.fun < 1e6:
            # Polish with Nelder-Mead
            res = minimize(_neg_log_score_from_params, 
                       x0=de_res_c4.x, 
                       method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000})
            
            r1, theta2, r2 = res.x
            pts1 = np.array([
                [r1 * cos(0), r1 * sin(0)],
                [r1 * cos(pi/2), r1 * sin(pi/2)],
                [r1 * cos(pi), r1 * sin(pi)],
                [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
            ])
            pts2 = np.array([
                [r2 * cos(theta2), r2 * sin(theta2)],
                [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
                [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
                [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
            ])
            candidate_pts = np.vstack([pts1, pts2])
            sc = _score(candidate_pts)
            if sc > best_score:
                best_score = sc
                best_pts = candidate_pts
    except Exception as e:
        pass
    
    # 2. Refine the best C4 configuration by allowing symmetry breaking in 16D space
    # Use the best C4 points as the initial guess for a local optimizer (SLSQP/Powell)
    if time.time() < deadline - 20.0 and best_pts is not None:
        
        def _neg_log_score_16d(pts):
            n = 8
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            diam = np.max(dists)
            if diam < 1e-12:
                return 1e10
            scale = 2.0 / diam
            dists_scaled = dists * scale
            mask = ~np.eye(n, dtype=bool)
            if np.any(dists_scaled[mask] < 1e-12):
                return 1e10
            log_prod = np.sum(np.log(dists_scaled[mask]))
            return n * log(n) - log_prod

        try:
            x0_16d = best_pts.flatten()
            
            # Use SLSQP for local refinement, allowing symmetry breaking
            # SLSQP can handle the smooth landscape well if started near a local min
            slsqp_res = minimize(_neg_log_score_16d, x0_16d, method='SLSQP', 
                                 options={'maxiter': 200, 'ftol': 1e-12})
            
            if slsqp_res.fun < 1e6:
                candidate_pts = slsqp_res.x.reshape(8, 2)
                sc = _score(candidate_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = candidate_pts
            
            # Also try Powell, which is derivative-free and robust
            if time.time() < deadline - 10.0:
                powell_res = minimize(_neg_log_score_16d, x0_16d, method='Powell', 
                                      options={'maxiter': 100, 'ftol': 1e-12})
                if powell_res.fun < 1e6:
                    candidate_pts = powell_res.x.reshape(8, 2)
                    sc = _score(candidate_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = candidate_pts
                        
        except Exception as e:
            pass

    # 3. Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
