"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Barycentric coordinates + Soft-Min Annealing + Nelder-Mead Polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

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

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    # Inverse of the linear transformation
    # x = 0.5*v + (1-u-v)*0.5 = 0.5 - 0.5*u
    # y = v*S3H + (1-u-v)*S3H = S3H - u*S3H
    # From y: u = (S3H - y) / S3H
    # From x: v = 1 - 2*x
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])

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
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we can use max subtraction, but areas are small.
    # areas are in range [0, 0.433]. beta is large.
    # max_a = np.max(areas)
    # shifted = -beta * (areas - max_a)
    # log_sum = np.log(np.sum(np.exp(shifted)))
    # soft_min = max_a - log_sum / beta
    
    # Simpler and more stable for small areas:
    # Since areas > 0, we can just compute directly if beta is not too huge.
    # But let's stick to the stable version.
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip shifted to avoid underflow issues in exp if beta is very large
    # exp(-large) -> 0. This is fine.
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
    # 1. Symmetric: 3 corners, 3 midpoints, 1 center, 4 others
    init_bary_list = []
    
    # Base symmetric configuration
    # Corners: (0,0), (1,0), (0,1) in barycentric (u,v,w) -> (u,v)
    # V0=(1,0,0) -> (1,0)
    # V1=(0,1,0) -> (0,1)
    # V2=(0,0,1) -> (0,0)
    # Midpoints:
    # V0-V1: (0.5, 0.5, 0) -> (0.5, 0.5)
    # V1-V2: (0, 0.5, 0.5) -> (0, 0.5)
    # V2-V0: (0.5, 0, 0.5) -> (0.5, 0)
    # Center: (1/3, 1/3, 1/3) -> (1/3, 1/3)
    
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],   # Extra 1
        [0.3, 0.3],   # Extra 2
        [0.4, 0.2],   # Extra 3
        [0.2, 0.4],   # Extra 4
    ])
    init_bary_list.append(base_sym)

    # Generate random initial configurations
    for _ in range(50):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))
    
    # Add some "near-boundary" configurations
    for _ in range(20):
        u = rng.random(N)
        v = rng.random(N)
        # Push some points to edges
        for i in range(N):
            if rng.random() < 0.3:
                # Push to one of the edges
                edge = rng.integers(0, 3)
                if edge == 0: # u=0
                    u[i] = 0
                elif edge == 1: # v=0
                    v[i] = 0
                else: # u+v=1
                    s = u[i] + v[i]
                    if s > 0:
                        u[i] /= s
                        v[i] /= s
                    else:
                        u[i] = 0.5
                        v[i] = 0.5
        init_bary_list.append(np.column_stack([u, v]))

    # Optimization Loop
    # We will iterate through initial configs, perform annealing, and keep the best.
    # To save time, we will limit the number of full annealing runs.
    max_starts = 60
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(30):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule for beta
        betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(20):
                break
            
            # Define objective in barycentric coords
            def objective(x):
                # x is flattened bary coords (2*N)
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum (exact min)
        if _elapsed_ok(10):
            # Nelder-Mead on flattened bary coords
            # We use a slightly perturbed version to escape local minima if needed
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                # Nelder-Mead is slow, so we limit iterations
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # If we have time, do a few random perturbations and re-optimize
        if _elapsed_ok(15) and current_start < max_starts // 2:
            for _ in range(2):
                if not _elapsed_ok(10):
                    break
                perturbed = current_bary + rng.normal(0, 0.05, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                # Quick L-BFGS-B polish
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, 500.0)
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-12})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    if new_bary is not None:
                        current_bary = new_bary
                except Exception:
                    pass

    # Final polish on the absolute best found
    if best_bary is not None and _elapsed_ok(5):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass

    if best_cart is None:
        # Fallback
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
