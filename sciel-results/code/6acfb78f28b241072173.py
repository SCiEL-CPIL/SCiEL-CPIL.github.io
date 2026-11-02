"""Heilbronn-11: Differential Evolution with Barycentric Parameterization and NM Polish."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 140  # safety margin from 150s limit

def _params_to_cart(params):
    """
    Convert parameters (x_i, y_i, s_i) for each point to Cartesian coordinates.
    Mapping:
    u = s * x
    v = (1 - s) * y
    w = 1 - u - v
    
    Since x, y, s in [0, 1]:
    u >= 0, v >= 0
    u + v = s*x + (1-s)*y <= s*1 + (1-s)*1 = 1
    So w >= 0.
    
    Cartesian:
    P = u*V0 + v*V1 + w*V2
    V0 = (0,0), V1 = (1,0), V2 = (0.5, S3H)
    P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H)
    """
    # params shape: (N, 3) -> (x, y, s)
    x = params[:, 0]
    y = params[:, 1]
    s = params[:, 2]
    
    u = s * x
    v = (1.0 - s) * y
    w = 1.0 - u - v
    
    # Ensure numerical stability for w
    w = np.maximum(w, 0.0)
    
    px = v + 0.5 * w
    py = w * S3H
    
    return np.column_stack((px, py))

def _min_area_cart(pts):
    """Compute the minimum triangle area for a set of Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _objective_de(params):
    """Objective function for Differential Evolution. Minimizes negative min area."""
    pts = _params_to_cart(params)
    area = _min_area_cart(pts)
    return -area

def _project_cartesian(pts):
    """Project points into the triangle (for safety in NM phase)."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    
    def _time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Bounds for DE: each point has 3 parameters (x, y, s) in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)
    
    best_area = 0.0
    best_params = None
    
    # Phase 1: Differential Evolution
    # popsize=20 for better exploration, maxiter=200 for speed
    # tol=1e-6 to stop early if convergence is reached
    try:
        res = differential_evolution(
            _objective_de,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True  # Let DE do its own polish first, but we will do a better one
        )
        
        if _time_left() > 10:
            best_params = res.x
            best_pts = _params_to_cart(best_params)
            best_area = _min_area_cart(best_pts)
            
    except Exception:
        # Fallback if DE fails
        best_params = np.random.rand(N, 3)
        best_pts = _params_to_cart(best_params)
        best_area = _min_area_cart(best_pts)

    # Phase 2: Nelder-Mead Polish on Cartesian Coordinates
    # We optimize in Cartesian space for the final polish as it's more direct
    if _time_left() > 5:
        current_pts = best_pts
        
        def nm_obj(x_flat):
            pts = x_flat.reshape(-1, 2)
            pts = _project_cartesian(pts)
            return -_min_area_cart(pts)
        
        # Run NM for a few seconds or until convergence
        try:
            res_nm = minimize(
                nm_obj,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if _time_left() > 0:
                candidate_pts = res_nm.x.reshape(-1, 2)
                candidate_pts = _project_cartesian(candidate_pts)
                candidate_area = _min_area_cart(candidate_pts)
                
                if candidate_area > best_area:
                    best_area = candidate_area
                    current_pts = candidate_pts
        except Exception:
            pass

    # Ensure final result is valid
    final_pts = _project_cartesian(current_pts)
    
    # One last check
    final_area = _min_area_cart(final_pts)
    if final_area > best_area:
        best_area = final_area
        
    return final_pts

# EVOLVE-BLOCK-END
