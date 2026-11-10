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

def _neg_log_score_16d(pts):
    """Negative log score for 8 free points."""
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
    return _neg_log_score_16d(pts)

def _two_squares_from_params(params):
    """
    Generate 8 points as two concentric, rotated regular squares.
    params = [r1, theta1, r2, theta2]
    Square 1: vertices at radius r1, starting angle theta1
    Square 2: vertices at radius r2, starting angle theta2
    """
    r1, theta1, r2, theta2 = params
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _neg_log_score_two_squares(params):
    """Negative log score for two-squares configuration."""
    pts = _two_squares_from_params(params)
    return _neg_log_score_16d(pts)

def _polish_16d(pts, deadline, max_time_per_method=5.0):
    """
    Apply multiple local optimization methods to polish 16D point configuration.
    Returns the best result found.
    """
    best_pts = pts.copy()
    best_val = _neg_log_score_16d(pts)
    
    methods = [
        ('SLSQP', {'maxiter': 200, 'ftol': 1e-14}),
        ('Nelder-Mead', {'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}),
        ('Powell', {'xtol': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'maxfev': 10000}),
    ]
    
    for method_name, opts in methods:
        if time.time() > deadline - 3.0:
            break
        try:
            def _obj(x_flat):
                return _neg_log_score_16d(x_flat.reshape(8, 2))
            
            res = minimize(_obj, best_pts.flatten(), method=method_name, options=opts)
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Final ultra-tight Nelder-Mead polish
    if time.time() < deadline - 2.0:
        try:
            def _obj(x_flat):
                return _neg_log_score_16d(x_flat.reshape(8, 2))
            
            res = minimize(_obj, best_pts.flatten(), method='Nelder-Mead',
                          options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 20000, 'maxfev': 20000})
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    best_score = 0.0
    best_pts = None
    
    # 1. C4 symmetric optimization (3 variables)
    try:
        res = minimize(_neg_log_score_from_params, 
                       x0=[1.0, pi/4, 1.0], 
                       method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000})
        
        if res.fun < 1e6:
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
    except Exception:
        pass
    
    # 2. Two concentric rotated squares with independent orientations (4 params)
    if time.time() < deadline - 40.0:
        try:
            bounds = [(0.1, 2.0), (0.0, 2.0*pi), (0.1, 2.0), (0.0, 2.0*pi)]
            
            de_res = differential_evolution(_neg_log_score_two_squares, bounds,
                                            maxiter=150, popsize=25, seed=42,
                                            tol=1e-10, polish=False, init='sobol')
            
            if de_res.fun < 1e6:
                candidate_pts = _two_squares_from_params(de_res.x)
                # Polish with Nelder-Mead
                nm_res = minimize(_neg_log_score_two_squares, de_res.x,
                                  method='Nelder-Mead',
                                  options={'xatol': 1e-12, 'fatol': 1e-12,
                                           'maxiter': 15000, 'maxfev': 15000})
                if nm_res.fun < 1e6 and nm_res.fun <= de_res.fun + 1e-6:
                    candidate_pts = _two_squares_from_params(nm_res.x)
                
                sc = _score(candidate_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = candidate_pts
        except Exception:
            pass
    
    # 3. Broader 16D DE search with multiple diverse seeds
    if time.time() < deadline - 25.0:
        bounds_16d = [(-2.0, 2.0)] * 16
        
        def _de_objective(x_flat):
            if time.time() > deadline - 5.0:
                return 1e10
            pts = x_flat.reshape(8, 2)
            return _neg_log_score_16d(pts)
        
        try:
            # Run DE with moderate settings
            de_res = differential_evolution(_de_objective, bounds_16d, maxiter=80, popsize=20, 
                                            seed=42, tol=1e-8, polish=False)
            
            if de_res.fun < 1e6:
                candidate_pts = de_res.x.reshape(8, 2)
                
                # Intensive multi-method polish
                if time.time() < deadline - 8.0:
                    polished_pts = _polish_16d(candidate_pts, deadline)
                    sc = _score(polished_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = polished_pts
                else:
                    sc = _score(candidate_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = candidate_pts
        except Exception:
            pass
    
    # 4. Multi-start local optimization from diverse initial points
    if time.time() < deadline - 15.0:
        # Generate diverse starting points
        start_points = []
        
        # Regular octagon
        start_points.append(np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]))
        
        # Two squares at different ratios
        for r_ratio in [0.5, 0.7, 1.0, 1.3]:
            angles1 = np.array([0, pi/2, pi, 3*pi/2])
            angles2 = pi/4 + np.array([0, pi/2, pi, 3*pi/2])
            pts1 = np.column_stack([1.0 * np.cos(angles1), 1.0 * np.sin(angles1)])
            pts2 = np.column_stack([r_ratio * np.cos(angles2), r_ratio * np.sin(angles2)])
            start_points.append(np.vstack([pts1, pts2]))
        
        # Random perturbations of regular octagon
        rng = np.random.RandomState(123)
        for _ in range(3):
            oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
            perturbed = oct_pts + rng.normal(0, 0.1, (8, 2))
            start_points.append(perturbed)
        
        for sp in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                polished_pts = _polish_16d(sp, deadline)
                sc = _score(polished_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = polished_pts
            except Exception:
                pass
    
    # 5. Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
