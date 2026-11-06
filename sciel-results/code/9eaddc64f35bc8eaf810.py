import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _score(pts):
    """
    Calculate the score for a set of points.
    Score = Delta / n^n, where Delta is the product of distances of all ordered pairs
    after scaling the set so that the diameter is 2.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    dia = d.max()
    if dia < 1e-12:
        return 0.0
    
    # Scale to diameter 2
    # d_scaled = d * (2.0 / dia)
    # Delta = prod_{i!=j} d_scaled_ij
    # log(Delta) = sum_{i!=j} (log(d_ij) + log(2) - log(dia))
    
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    log_prod_scaled = np.log(d_safe).sum() + _N * (_N - 1) * (math.log(2.0) - math.log(dia))
    log_score = log_prod_scaled - _N * math.log(_N)
    
    # Clip to avoid overflow/underflow in exp if needed, though scores are usually near 1
    if log_score < -700: return 0.0
    if log_score > 700: return math.inf
    
    return math.exp(log_score)

def _get_points_4fold(params):
    """
    Generate 8 points with 4-fold rotational symmetry.
    params: [theta0, r1, theta1]
    r0 is fixed to 1.0 (scale is invariant, but fixing one radius helps optimization).
    Actually, let's use 4 parameters: [r0, theta0, r1, theta1] to allow more flexibility,
    or keep r0=1 and rely on the fact that scaling is invariant?
    Wait, if we fix r0=1, the diameter is determined by the configuration.
    The score is scale invariant.
    Let's use 4 parameters: [r0, theta0, r1, theta1]
    Points:
    P0 = (r0, 0)  -- wait, let's put them in polar coordinates relative to center?
    No, the symmetry is around the origin.
    
    Let's define the orbit of a point (x, y) under 90 deg rotation.
    Point 1: (x1, y1)
    Point 2: (-y1, x1)
    Point 3: (-x1, -y1)
    Point 4: (y1, -x1)
    
    Point 5: (x2, y2)
    Point 6: (-y2, x2)
    Point 7: (-x2, -y2)
    Point 8: (y2, -x2)
    
    Params: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = params
    
    p1 = np.array([x1, y1])
    p2 = np.array([-y1, x1])
    p3 = np.array([-x1, -y1])
    p4 = np.array([y1, -x1])
    
    p5 = np.array([x2, y2])
    p6 = np.array([-y2, x2])
    p7 = np.array([-x2, -y2])
    p8 = np.array([y2, -x2])
    
    return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

def _objective_4fold(params):
    pts = _get_points_4fold(params)
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
    s = _score(pts)
    return -s

def _refine_generic(pts, maxiter=2000):
    """
    Refine a full 8-point configuration using L-BFGS-B.
    """
    def obj(x):
        p = x.reshape(_N, 2)
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return 1e12
        return -_score(p)
        
    try:
        res = minimize(obj, pts.ravel(), method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
        cand = res.x.reshape(_N, 2)
        d = np.linalg.norm(cand[:, np.newaxis, :] - cand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
             return pts, _score(pts)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)

def _refine_4fold(params, maxiter=2000):
    """
    Refine parameters of the 4-fold symmetric configuration.
    """
    try:
        res = minimize(_objective_4fold, params, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
        cand_params = res.x
        pts = _get_points_4fold(cand_params)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
             return params, _score(_get_points_4fold(params))
        return cand_params, _score(pts)
    except Exception:
        return params, _score(_get_points_4fold(params))

def construct():
    t0 = time.time()
    best_pts = None
    best_score = 0.0
    best_params_4fold = None
    
    # 1. Global search in 4-fold symmetric space (4 variables: x1, y1, x2, y2)
    # Bounds: points should be reasonable. Diameter will be normalized later.
    # Let's search in [-5, 5] for coordinates.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # Differential Evolution
    try:
        if _remaining() > 60:
            res_de = differential_evolution(
                _objective_4fold, bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                popsize=15
            )
            if res_de.fun < 1e10:
                best_params_4fold = res_de.x
                pts = _get_points_4fold(best_params_4fold)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
    except Exception as e:
        pass

    # 2. Refine the best 4-fold params
    if best_params_4fold is not None and _remaining() > 30:
        refined_params, s = _refine_4fold(best_params_4fold, maxiter=1000)
        if s > best_score:
            best_score = s
            best_params_4fold = refined_params
            best_pts = _get_points_4fold(refined_params)

    # 3. Try multiple random starts in 4-fold space and refine
    rng = np.random.default_rng(12345)
    num_random_starts = 30
    for i in range(num_random_starts):
        if _remaining() < 20: break
        
        # Random parameters
        # Sample from a disk or square
        # Let's sample x1, y1, x2, y2 in [-2, 2] to avoid too large initial configs
        params = rng.uniform(-2.0, 2.0, size=4)
        
        # Quick local refinement
        refined_params, s = _refine_4fold(params, maxiter=500)
        if s > best_score:
            best_score = s
            best_params_4fold = refined_params
            best_pts = _get_points_4fold(refined_params)

    # 4. If we have time, try to break symmetry by refining the full 16-variable configuration
    # Start from the best 4-fold symmetric configuration
    if best_pts is not None and _remaining() > 40:
        # Try a few perturbations to break symmetry
        for i in range(5):
            if _remaining() < 20: break
            # Small noise
            noisy = best_pts + rng.normal(0, 0.1, size=(8, 2))
            cand, s = _refine_generic(noisy, maxiter=500)
            if s > best_score:
                best_score = s
                best_pts = cand

    # 5. Fallback to regular octagon if nothing worked
    if best_pts is None:
        best_pts = np.array([[math.cos(2*math.pi*k/8), math.sin(2*math.pi*k/8)] for k in range(8)])
        best_score = _score(best_pts)

    # Final check and cleanup
    best_pts = np.asarray(best_pts, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-5:
         for i in range(_N):
             for j in range(i+1, _N):
                 if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-5:
                     best_pts[j] += [1e-5, 0]
                     
    return [[float(x), float(y)] for (x, y) in best_pts]
