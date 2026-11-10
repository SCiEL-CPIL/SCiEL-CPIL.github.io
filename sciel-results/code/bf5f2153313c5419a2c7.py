"""Program to maximize product of distances for 8 planar points at fixed diameter."""
import numpy as np
import time
from math import cos, sin, pi, log, sqrt, exp
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
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

def _two_squares_from_params(params):
    """
    Generate 8 points as two concentric, rotated regular squares.
    params = [r1, theta1, r2, theta2]
    Square 1: vertices at radius r1, starting angle theta1
    Square 2: vertices at radius r2, starting angle theta2
    """
    r1, theta1, r2, theta2 = params
    if r1 < 0.01 or r2 < 0.01:
        return None
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _neg_log_score_two_squares(params):
    """Negative log score for two-squares configuration."""
    pts = _two_squares_from_params(params)
    if pts is None:
        return 1e10
    return _neg_log_score_16d(pts)

def _six_param_model(params):
    """
    6-parameter model: two concentric rotated squares with independent radii, orientations,
    and independent x/y offsets for each square.
    Square 1 center: (x_off, y_off)
    Square 2 center: (-x_off, -y_off)
    """
    r1, r2, theta1, theta2, x_off, y_off = params
    if r1 < 0.01 or r2 < 0.01:
        return None
    
    # Square 1
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * cos(angles1) + x_off, r1 * sin(angles1) + y_off])
    
    # Square 2
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts2 = np.column_stack([r2 * cos(angles2) - x_off, r2 * sin(angles2) - y_off])
    
    return np.vstack([pts1, pts2])

def _six_param_neglog(params):
    """Negative log score for 6-parameter model."""
    pts = _six_param_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score_16d(pts)

def _c4_symmetric_model(params):
    """
    C4 symmetric configuration:
    params = [r1, theta2, r2]
    Orbit 1: r1 * exp(i * (0, pi/2, pi, 3pi/2))
    Orbit 2: r2 * exp(i * (theta2, theta2 + pi/2, ...))
    """
    r1 = params[0]
    theta2 = params[1]
    r2 = params[2]
    if r1 < 0.01 or r2 < 0.01:
        return None
    
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
    
    return np.vstack([pts1, pts2])

def _neg_log_score_c4(params):
    """Negative log score for C4 symmetric configuration."""
    pts = _c4_symmetric_model(params)
    if pts is None:
        return 1e10
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
        ('Powell', {'xtol': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'maxfev': 10000}),
        ('Nelder-Mead', {'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}),
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
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    # === PHASE 1: C4 symmetric optimization (3 variables) ===
    if time.time() < deadline - 100:
        try:
            res = minimize(_neg_log_score_c4, 
                           x0=[1.0, pi/4, 1.0], 
                           method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000})
            
            if res.fun < 1e6:
                candidate_pts = _c4_symmetric_model(res.x)
                if candidate_pts is not None:
                    sc = _score(candidate_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # === PHASE 2: Two concentric rotated squares with independent orientations (4 params) ===
    if time.time() < deadline - 60:
        try:
            bounds = [(0.1, 2.0), (0.0, 2.0*pi), (0.1, 2.0), (0.0, 2.0*pi)]
            
            de_res = differential_evolution(_neg_log_score_two_squares, bounds,
                                            maxiter=150, popsize=25, seed=42,
                                            tol=1e-10, polish=False, init='sobol')
            
            if de_res.fun < 1e6:
                candidate_pts = _two_squares_from_params(de_res.x)
                if candidate_pts is not None:
                    # Polish with Nelder-Mead
                    nm_res = minimize(_neg_log_score_two_squares, de_res.x,
                                      method='Nelder-Mead',
                                      options={'xatol': 1e-12, 'fatol': 1e-12,
                                               'maxiter': 15000, 'maxfev': 15000})
                    if nm_res.fun < 1e6 and nm_res.fun <= de_res.fun + 1e-6:
                        candidate_pts = _two_squares_from_params(nm_res.x)
                    
                    if candidate_pts is not None:
                        sc = _score(candidate_pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # === PHASE 3: 6-parameter model with offsets ===
    if time.time() < deadline - 40:
        try:
            bounds = [(0.1, 2.0), (0.1, 2.0), (0, pi/2), (0, pi/2), (-1.0, 1.0), (-1.0, 1.0)]
            
            de_result = differential_evolution(
                _six_param_neglog, bounds,
                seed=42, maxiter=200, popsize=25,
                tol=1e-10, polish=True,
                workers=1
            )
            if de_result.fun < 1e6:
                pts = _six_param_model(de_result.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # Try a second DE run with different seed if time permits
    if time.time() < deadline - 25:
        try:
            bounds = [(0.1, 2.0), (0.1, 2.0), (0, pi/2), (0, pi/2), (-1.0, 1.0), (-1.0, 1.0)]
            
            de_result2 = differential_evolution(
                _six_param_neglog, bounds,
                seed=123, maxiter=150, popsize=20,
                tol=1e-10, polish=True,
                workers=1
            )
            if de_result2.fun < 1e6:
                pts = _six_param_model(de_result2.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass

    # === PHASE 4: Local optimization on full 16-dim space ===
    # We use the best point found on the manifold as a starting point for full optimization.
    # This allows breaking the square symmetry if it helps.
    
    def objective(x):
        pts_x = x.reshape(n, 2)
        return _neg_log_score_16d(pts_x)
    
    # Constraint: diameter <= 2
    def constraint_diameter(x):
        pts_x = x.reshape(n, 2)
        diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return 2.0 - np.max(dists)
    
    constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
    
    # Initial optimization from best_pts
    if time.time() < deadline - 15:
        try:
            result = minimize(objective, best_pts.flatten(), method='SLSQP',
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # L-BFGS-B from best_pts (unconstrained, but usually converges to similar if scaled well)
        try:
            result = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # === PHASE 5: Perturbation-based polish in full space ===
    rng = np.random.RandomState(42)
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            noise = rng.randn(n, 2) * 0.02
            perturbed = best_pts + noise
            try:
                result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                constraints=constraints, options={'maxiter': 500, 'ftol': 1e-15})
                if result.fun < 1e6:
                    opt_pts = result.x.reshape(n, 2)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final multi-method polish
    if time.time() < deadline - 2:
        best_pts = _polish_16d(best_pts, deadline)
        final_score = _score(best_pts)
        if final_score > best_score:
            best_score = final_score

    # Final check for coordinate bounds
    if np.any(np.abs(best_pts) >= 1e6):
        best_pts = oct_pts.copy()

    return best_pts.tolist()
