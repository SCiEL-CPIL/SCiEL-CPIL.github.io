"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _log_score_vec(pts):
    """
    Fast vectorized computation of log(Delta / n^n) after scaling to diameter 2.
    pts: numpy array of shape (8, 2)
    Returns the log of the score (higher is better).
    """
    n = pts.shape[0]
    if n < 2:
        return -1e10
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Mask diagonal (self-distances)
    np.fill_diagonal(d2, 1.0)  # temporary, will be excluded
    d = np.sqrt(d2)
    # Diameter = max pairwise distance
    D = np.max(d)
    if D < 1e-12:
        return -1e10
    s = 2.0 / D
    # Log of product of all ordered-pair distances after scaling
    d_scaled = d * s
    np.fill_diagonal(d_scaled, 1.0)
    d_scaled = np.clip(d_scaled, 1e-15, None)
    log_p = np.sum(np.log(d_scaled))
    log_score = log_p - n * math.log(n)
    return log_score

def _two_squares(r1, r2, theta1, theta2, dy):
    """
    Generate 8 points as two concentric, rotated regular squares.
    Square 1: radius r1, rotation theta1, centered at (0, dy)
    Square 2: radius r2, rotation theta2, centered at (0, 0)
    """
    # Square 1 vertices (rotated by theta1)
    a1 = np.array([0, math.pi/2, math.pi, 3*math.pi/2]) + theta1
    sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    sq1[:, 1] += dy  # vertical offset
    
    # Square 2 vertices (rotated by theta2)
    a2 = np.array([0, math.pi/2, math.pi, 3*math.pi/2]) + theta2
    sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    
    return np.vstack([sq1, sq2])

def _param_objective(params):
    """
    Objective for the 5-parameter two-squares model.
    params: [r1, r2, theta1, theta2, dy]
    """
    r1, r2, theta1, theta2, dy = params
    # Ensure positive radii
    r1 = max(abs(r1), 1e-10)
    r2 = max(abs(r2), 1e-10)
    pts = _two_squares(r1, r2, theta1, theta2, dy)
    ls = _log_score_vec(pts)
    return -ls

def _objective(pts_flat):
    """
    Objective for scipy minimization (we minimize negative log score).
    pts_flat: array of 16 floats (8 points x 2 coords)
    """
    pts = pts_flat.reshape(8, 2)
    ls = _log_score_vec(pts)
    return -ls

def _score_from_pts(pts_list):
    """Compute the actual score for a list of [x,y] points."""
    pts = np.array(pts_list, dtype=float)
    ls = _log_score_vec(pts)
    return math.exp(ls)

def construct():
    """
    Find 8 distinct points maximizing the product of distances at fixed diameter.
    Uses two-squares parameterization with differential evolution + local polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def regular_octagon(radius=1.0):
        angles = 2 * math.pi * np.arange(8) / 8
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    best_pts = regular_octagon()
    best_log_score = _log_score_vec(best_pts)

    # Phase 1: Differential evolution on 5-parameter two-squares model
    # params: [r1, r2, theta1, theta2, dy]
    bounds = [
        (0.1, 2.0),   # r1
        (0.1, 2.0),   # r2
        (-math.pi/4, math.pi/4),  # theta1
        (-math.pi/4, math.pi/4),  # theta2
        (-1.0, 1.0),  # dy
    ]
    
    # Seed DE with regular octagon equivalent: r1=r2=1, theta1=0, theta2=pi/4, dy=0
    # But we need to be careful: two squares with same radius and 45 deg offset = regular octagon
    x0_de = [1.0, 1.0, 0.0, math.pi/4, 0.0]
    
    try:
        de_res = differential_evolution(
            _param_objective, bounds,
            seed=42, maxiter=300, tol=1e-12,
            popsize=15, init='sobol',
            polish=False,  # We'll do our own polish
            updating='immediate',
            workers=1
        )
        if de_res.fun < -best_log_score:
            de_params = de_res.x
            pts_de = _two_squares(*de_params)
            ls_de = _log_score_vec(pts_de)
            if ls_de > best_log_score:
                best_log_score = ls_de
                best_pts = pts_de
    except Exception:
        pass

    if time.time() > deadline - 20.0:
        # Not enough time for local polish, skip to output
        pass
    else:
        # Phase 2: Local polish on the 5-parameter model
        if 'de_res' in dir() and de_res.success:
            start_params = de_res.x
        else:
            start_params = [1.0, 1.0, 0.0, math.pi/4, 0.0]
        
        # Try multiple local optimizations from the DE result
        for method in ['Powell', 'Nelder-Mead']:
            try:
                res_local = minimize(
                    _param_objective, start_params,
                    method=method,
                    options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12,
                             'adaptive': method == 'Nelder-Mead'}
                )
                if res_local.fun < -best_log_score:
                    params_local = res_local.x
                    pts_local = _two_squares(*params_local)
                    ls_local = _log_score_vec(pts_local)
                    if ls_local > best_log_score:
                        best_log_score = ls_local
                        best_pts = pts_local
                        start_params = params_local  # Update for next method
            except Exception:
                pass

    # Phase 3: Full 16-DOF polish from the best symmetric result
    if time.time() < deadline - 10.0:
        init_flat = best_pts.flatten()
        
        # Powell on full 16-DOF
        try:
            res_full = minimize(
                _objective, init_flat,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res_full.fun < -best_log_score:
                pts_full = res_full.x.reshape(8, 2)
                ls_full = _log_score_vec(pts_full)
                if ls_full > best_log_score:
                    best_log_score = ls_full
                    best_pts = pts_full
        except Exception:
            pass

        # Nelder-Mead polish from Powell result
        if time.time() < deadline - 5.0:
            try:
                start_nm = res_full.x if 'res_full' in dir() else init_flat
                res_nm = minimize(
                    _objective, start_nm,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < -best_log_score:
                    pts_nm = res_nm.x.reshape(8, 2)
                    ls_nm = _log_score_vec(pts_nm)
                    if ls_nm > best_log_score:
                        best_log_score = ls_nm
                        best_pts = pts_nm
            except Exception:
                pass

    # Convert back to list of [x, y]
    best_list = [[float(p[0]), float(p[1])] for p in best_pts]
    
    # Ensure distinctness
    for a in range(8):
        for b in range(a + 1, 8):
            d = math.hypot(best_list[a][0] - best_list[b][0], best_list[a][1] - best_list[b][1])
            if d < 1e-6:
                best_list[b][0] += 1e-5 * (b - a + 1)
                best_list[b][1] += 1e-5 * (b - a + 2)
    
    # Ensure coordinate bounds
    for p in best_list:
        if abs(p[0]) > 1e6:
            p[0] = math.copysign(1e6 - 1, p[0])
        if abs(p[1]) > 1e6:
            p[1] = math.copysign(1e6 - 1, p[1])
    
    return best_list
# EVOLVE-BLOCK-END
