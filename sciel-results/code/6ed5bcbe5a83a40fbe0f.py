import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _two_squares_score(r_outer, r_inner, delta_angle):
    """Score for two concentric squares with given parameters."""
    # Outer square: 4 points at radius r_outer, angles 0, pi/2, pi, 3pi/2
    # Inner square: 4 points at radius r_inner, angles delta_angle, delta_angle+pi/2, delta_angle+pi, delta_angle+3pi/2
    outer_angles = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])
    inner_angles = outer_angles + delta_angle
    
    pts_outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    pts_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    pts = np.vstack([pts_outer, pts_inner])
    return _score(pts)

def _two_squares_objective(params):
    """Objective for optimizing two squares parameters."""
    r_outer, r_inner, delta_angle = params
    if r_outer < 0.01 or r_inner < 0.0:
        return 1e12
    return -_two_squares_score(r_outer, r_inner, delta_angle)

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())

def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    # Sum of logs for i < j, multiply by 2 for ordered pairs
    tri_idx = np.triu_indices(_N, 1)
    return 2.0 * float(np.sum(np.log(d[tri_idx])))

def _score_from_log_product_and_diameter(lp, dia):
    if dia <= 1e-12:
        return 0.0
    n = _N
    # Score = Delta / n^n
    # Delta = prod_{i!=j} d_ij_scaled
    # d_scaled = d * (2/dia)
    # Delta = (2/dia)^(n(n-1)) * prod_{i!=j} d_ij
    # log Score = n(n-1) * log(2/dia) + log(prod d_ij) - n log n
    # log(prod d_ij) = lp (which is sum of logs for ordered pairs)
    # Wait, _log_product returns 2 * sum(i<j). This is sum(i!=j). Correct.
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))

def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)

def _objective(pts):
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)

def _refine(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: L-BFGS-B
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High precision
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial guess: Regular Octagon
    best = None
    best_score = 0.0
    
    # 1. Try regular octagon
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    cand, s = _refine(oct_pts)
    best, best_score = cand, s

    # 2. Try radial deformations of the octagon
    # Parameterize by radii r_k, fixed angles theta_k = 2*pi*k/8
    # This exploits the symmetry of the local max and allows escaping it radially.
    
    def radial_objective(radii):
        radii = np.asarray(radii, dtype=float)
        thetas = 2 * math.pi * np.arange(_N) / _N
        pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
        return _objective(pts)

    # Optimize radii
    if _remaining() > 10.0:
        r0 = np.ones(_N)
        # Bounds: radii should be positive
        bounds = [(0.1, 2.0)] * _N
        try:
            res_r = minimize(radial_objective, r0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            radii_opt = res_r.x
            thetas = 2 * math.pi * np.arange(_N) / _N
            pts_radial = np.column_stack([radii_opt * np.cos(thetas), radii_opt * np.sin(thetas)])
            cand, s = _refine(pts_radial)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 3. Basin Hopping on the full 16D space, starting from best
    if _remaining() > 20.0:
        def bh_objective(x):
            return _objective(x.reshape(_N, 2))
        
        try:
            bh_res = basinhopping(
                bh_objective, best.ravel(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10}},
                niter=100, seed=12345, 
                stepsize=0.5,
                T=100.0,
                accept_test=lambda x, fx, x0, fx0: fx < fx0
            )
            cand = bh_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 4. Try "Butterfly" or "Two Squares" configurations
    # Configuration: 4 points on outer circle, 4 on inner circle
    if _remaining() > 10.0:
        pts_butterfly = []
        # Outer square
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts_butterfly.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        # Inner square
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts_butterfly.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        pts_butterfly = np.array(pts_butterfly, dtype=float)
        cand, s = _refine(pts_butterfly)
        if s > best_score:
            best, best_score = cand, s

    # 5. Two concentric squares: 3D grid search + local refinement
    if _remaining() > 15.0:
        # Grid search over (r_outer, r_inner, delta_angle)
        # r_outer in [0.5, 1.5], r_inner in [0.0, 1.0], delta_angle in [0, pi/2]
        r_outer_grid = np.linspace(0.5, 1.5, 8)
        r_inner_grid = np.linspace(0.0, 1.0, 8)
        delta_grid = np.linspace(0.0, math.pi/2, 8)
        
        best_3d_score = -1e18
        best_3d_params = None
        
        for ro in r_outer_grid:
            for ri in r_inner_grid:
                for da in delta_grid:
                    s = _two_squares_score(ro, ri, da)
                    if s > best_3d_score:
                        best_3d_score = s
                        best_3d_params = (ro, ri, da)
        
        # Local refinement from best grid point
        if best_3d_params is not None and _remaining() > 5.0:
            try:
                res_3d = minimize(_two_squares_objective, best_3d_params, method='L-BFGS-B',
                                  bounds=[(0.1, 2.0), (0.0, 1.5), (0.0, math.pi/2)],
                                  options={'maxiter': 500, 'ftol': 1e-12})
                ro_opt, ri_opt, da_opt = res_3d.x
                # Build the 8 points
                outer_angles = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])
                inner_angles = outer_angles + da_opt
                pts_outer = np.column_stack([ro_opt * np.cos(outer_angles), ro_opt * np.sin(outer_angles)])
                pts_inner = np.column_stack([ri_opt * np.cos(inner_angles), ri_opt * np.sin(inner_angles)])
                pts_3d = np.vstack([pts_outer, pts_inner])
                cand, s = _refine(pts_3d)
                if s > best_score:
                    best, best_score = cand, s
            except Exception:
                pass
    
    # 6. Random restarts if time remains
    while _remaining() > 10.0:
        # Random points in unit disk
        pts_rand = rng.uniform(-1.0, 1.0, size=(8, 2))
        cand, s = _refine(pts_rand)
        if s > best_score:
            best, best_score = cand, s

    # Final polish
    if _remaining() > 2.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s
    
    # Additional high-precision polish on the final best
    if _remaining() > 1.0:
        try:
            res_final = minimize(_objective, best.ravel(), method='SLSQP',
                                options={'maxiter': 500, 'ftol': 1e-14})
            cand = res_final.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    best = np.asarray(best, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]
                
    return [[float(x), float(y)] for (x, y) in best]
