"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 140  # Safety margin for 150s limit


def _bary_to_cart(bc):
    """Convert barycentric (b, c) to cartesian. a = 1-b-c.
    Vertices: A=(0,0), B=(1,0), C=(0.5, S3H)
    cart = b*B + c*C = (b + 0.5*c, S3H*c)
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack([x, y])


def _project_bary(bc):
    """Project barycentric coords to valid simplex: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    # Clip non-negative
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    # Clip b+c <= 1
    s = b + c
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        b[mask] *= scale
        c[mask] *= scale
    return np.column_stack([b, c])


def _min_area_bary(bc):
    """Min triangle area given barycentric coords."""
    cart = _bary_to_cart(bc)
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def construct():
    start_time = time.time()
    
    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    # Objective function for differential_evolution
    # Input: flat array of 22 values (b1, c1, ..., b11, c11)
    # Output: -min_area (since DE minimizes)
    def objective(flat_bc):
        bc = flat_bc.reshape(-1, 2)
        bc = _project_bary(bc)
        area = _min_area_bary(bc)
        return -area

    # Bounds for differential_evolution
    # b in [0, 1], c in [0, 1]
    # Note: We project to simplex inside objective, so bounds can be loose [0,1]
    bounds = [(0, 1)] * (2 * N)

    best_bc = None
    best_area = 0.0

    # Fallback: A known good symmetric configuration
    # This is a 3x3 grid-like structure plus vertices, adjusted for triangle
    # Let's use a simple symmetric start
    init_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.5, S3H/2], # Center
    ], dtype=float)
    
    # Convert to barycentric for DE initialization
    # c = y / S3H
    # b = x - 0.5 * c
    c_init = init_cart[:, 1] / S3H
    b_init = init_cart[:, 0] - 0.5 * c_init
    init_bc = np.column_stack([b_init, c_init])
    init_bc = _project_bary(init_bc)
    
    current_best_area = _min_area_bary(init_bc)
    best_bc = init_bc.copy()
    best_area = current_best_area

    # Run Differential Evolution
    # Keep it fast: small popsize, few iterations
    if _elapsed_ok(10):
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=8,
                tol=1e-4,
                seed=42,
                polish=True,
                update_interval=0,
                callback=None
            )
            
            if result.success or result.fun < 0:
                res_bc = _project_bary(result.x.reshape(-1, 2))
                res_area = _min_area_bary(res_bc)
                if res_area > best_area:
                    best_area = res_area
                    best_bc = res_bc
        except Exception:
            pass

    # If DE didn't help much, try a quick local polish on the best
    if _elapsed_ok(10) and best_bc is not None:
        from scipy.optimize import minimize
        
        def obj_local(flat_bc):
            bc_new = _project_bary(flat_bc.reshape(-1, 2))
            return -_min_area_bary(bc_new)
            
        try:
            res = minimize(
                obj_local,
                best_bc.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            new_bc = _project_bary(res.x.reshape(-1, 2))
            new_area = _min_area_bary(new_bc)
            if new_area > best_area:
                best_area = new_area
                best_bc = new_bc
        except Exception:
            pass

    # Final check and return
    if best_bc is None:
        # Absolute fallback
        return init_cart

    return _bary_to_cart(best_bc)
