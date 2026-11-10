"""Program to maximize the product of distances for 8 planar points with fixed diameter."""
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

def _neg_log_score(pts):
    """Negative log of the score for minimization."""
    n = len(pts)
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

def _centrally_symmetric_model(params):
    """
    Generate 8 points as 4 unique points and their negations (central symmetry).
    params = 8 floats representing 4 points (x1,y1, x2,y2, x3,y3, x4,y4)
    The 8 points are: p1, p2, p3, p4, -p1, -p2, -p3, -p4
    """
    pts_half = params.reshape(4, 2)
    pts = np.vstack([pts_half, -pts_half])
    return pts

def _centrally_symmetric_neglog(params):
    """Negative log score for centrally symmetric model."""
    pts = _centrally_symmetric_model(params)
    return _neg_log_score(pts)

def _two_squares_model(params):
    """
    Generate 8 points: two concentric squares.
    params = [r1, r2, theta1, theta2]
    Square 1: 4 points at angles theta1 + k*pi/2 with radius r1
    Square 2: 4 points at angles theta2 + k*pi/2 with radius r2
    """
    r1, r2, theta1, theta2 = params
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1)])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2)])
    return np.vstack([pts1, pts2])

def _two_squares_neglog(params):
    """Negative log score for two-squares model."""
    pts = _two_squares_model(params)
    return _neg_log_score(pts)

def _five_param_model(params):
    """
    5-parameter model: two concentric rotated squares with independent radii, orientations, and vertical offset.
    params = [r1, r2, theta1, theta2, y_offset]
    Square 1: 4 points at angles theta1 + k*pi/2 with radius r1, shifted by (0, y_offset)
    Square 2: 4 points at angles theta2 + k*pi/2 with radius r2, shifted by (0, -y_offset)
    """
    r1, r2, theta1, theta2, y_off = params
    if r1 < 0.01 or r2 < 0.01:
        return None
    
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1) + y_off])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2) - y_off])
    
    return np.vstack([pts1, pts2])

def _five_param_neglog(params):
    """Negative log score for 5-parameter model."""
    pts = _five_param_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    # === PHASE 1: Differential evolution on centrally symmetric 8D model ===
    if time.time() < deadline - 60:
        # Bounds for 4 points: each coordinate in [-2, 2]
        bounds_cs = [(-2.0, 2.0)] * 8
        
        try:
            de_res_cs = differential_evolution(
                _centrally_symmetric_neglog, bounds_cs,
                seed=42, maxiter=100, popsize=20,
                tol=1e-10, polish=True
            )
            if de_res_cs.fun < 1e6:
                pts = _centrally_symmetric_model(de_res_cs.x)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 2: Differential evolution on 5-parameter model ===
    if time.time() < deadline - 50:
        bounds_5p = [(0.1, 3.0), (0.1, 3.0), (0, pi/2), (0, pi/2), (-1.0, 1.0)]
        
        try:
            de_res_5p = differential_evolution(
                _five_param_neglog, bounds_5p,
                seed=42, maxiter=150, popsize=20,
                tol=1e-10, polish=True
            )
            if de_res_5p.fun < 1e6:
                pts = _five_param_model(de_res_5p.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 3: Differential evolution on 4-parameter two-squares model ===
    if time.time() < deadline - 40:
        bounds_2s = [(0.1, 3.0), (0.1, 3.0), (0, pi/2), (0, pi/2)]
        
        try:
            de_res_2s = differential_evolution(
                _two_squares_neglog, bounds_2s,
                seed=42, maxiter=100, popsize=20,
                tol=1e-10, polish=True
            )
            if de_res_2s.fun < 1e6:
                pts = _two_squares_model(de_res_2s.x)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 4: Local optimization in full 16D space from best found ===
    if time.time() < deadline - 30 and best_pts is not None:
        def objective_16d(x):
            pts_x = x.reshape(n, 2)
            return _neg_log_score(pts_x)
        
        # SLSQP with diameter constraint
        def constraint_diameter(x):
            pts_x = x.reshape(n, 2)
            diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            return 2.0 - np.max(dists)
        
        constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
        
        try:
            result = minimize(objective_16d, best_pts.flatten(), method='SLSQP',
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # L-BFGS-B
        try:
            result = minimize(objective_16d, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # === PHASE 5: Additional DE runs with different seeds ===
    if time.time() < deadline - 20:
        # Try different seeds on centrally symmetric model
        for seed in [123, 456]:
            if time.time() > deadline - 10:
                break
            bounds_cs = [(-2.0, 2.0)] * 8
            try:
                de_res = differential_evolution(
                    _centrally_symmetric_neglog, bounds_cs,
                    seed=seed, maxiter=80, popsize=15,
                    tol=1e-10, polish=True
                )
                if de_res.fun < 1e6:
                    pts = _centrally_symmetric_model(de_res.x)
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # === PHASE 6: Perturbation-based polish ===
    rng = np.random.RandomState(42)
    if time.time() < deadline - 10:
        def objective_16d(x):
            pts_x = x.reshape(n, 2)
            return _neg_log_score(pts_x)
        
        def constraint_diameter(x):
            pts_x = x.reshape(n, 2)
            diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            return 2.0 - np.max(dists)
        
        constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
        
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            noise = rng.randn(n, 2) * 0.05
            perturbed = best_pts + noise
            try:
                result = minimize(objective_16d, perturbed.flatten(), method='SLSQP',
                                constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
                if result.fun < 1e6:
                    opt_pts = result.x.reshape(n, 2)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # === PHASE 7: Fine polish with smaller perturbations ===
    if time.time() < deadline - 3:
        def objective_16d(x):
            pts_x = x.reshape(n, 2)
            return _neg_log_score(pts_x)
        
        for step_size in [0.01, 0.005, 0.002]:
            if time.time() > deadline - 2:
                break
            for trial in range(20):
                if time.time() > deadline - 1:
                    break
                noise = rng.randn(n, 2) * step_size
                perturbed = best_pts + noise
                try:
                    result = minimize(objective_16d, perturbed.flatten(), method='Powell',
                                    options={'maxiter': 100, 'ftol': 1e-14})
                    if result.fun < 1e6:
                        opt_pts = result.x.reshape(n, 2)
                        sc = _score(opt_pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    # Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
