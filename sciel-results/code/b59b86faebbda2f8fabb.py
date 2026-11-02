import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 145  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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

def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Stable soft min
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid underflow/overflow in exp if needed, but areas are small
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    # Enforce u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        # Scale down u and v to fit
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial configurations
    init_bary_list = []
    
    # 1. Symmetric grid-like configuration
    # Corners
    init_bary_list.append(np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [0.5, 0.5], [0.0, 0.5], [0.5, 0.0],
        [1/3, 1/3],
        [0.2, 0.2], [0.3, 0.3], [0.4, 0.2], [0.2, 0.4]
    ]))

    # 2. Random starts
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))

    # Optimization Loop
    max_starts = 40
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(30):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule
        betas = [20.0, 100.0, 500.0, 2000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(20):
                break
            
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum
        if _elapsed_ok(10):
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass

    # Final polish
    if best_bary is not None and _elapsed_ok(5):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass

    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart
