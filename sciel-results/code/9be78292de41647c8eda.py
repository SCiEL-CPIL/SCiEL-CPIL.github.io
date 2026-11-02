"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Barycentric coordinates + Soft-Min Annealing with analytic gradient + Nelder-Mead Polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import math

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

DEADLINE = 148  # Safety margin from 150s limit

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
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _soft_min_area_and_grad_cart(pts, beta):
    """Soft minimum of triangle areas and its gradient w.r.t. pts (Cartesian).
    Returns (soft_min_value, grad) where grad has shape (N, 2).
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min = max_a - log_sum / beta
    
    # weights w_i = exp(-beta * (a_i - max_a)) / sum_exp
    w = exp_shifted / sum_exp  # shape (M,)
    
    # d(soft_min)/d(pts) = -sum_i w_i * d(a_i)/d(pts)
    # a_i = 0.5 * |cross_i|
    # d(a_i)/d(pts) depends on sign of cross_i
    # cross = (pj - pi) x (pk - pi)
    # d(cross)/d(pi) = -dy2, dx2  (for x,y components)
    # d(cross)/d(pj) = dy2, -dx2
    # d(cross)/d(pk) = -dy1, dx1
    # d(a)/d(pt) = 0.5 * sign(cross) * d(cross)/d(pt)
    
    sign_cross = np.sign(cross)
    # Handle zero-cross: use 1.0 to avoid NaN
    sign_cross = np.where(np.abs(cross) < 1e-15, 1.0, sign_cross)
    
    grad = np.zeros((N, 2))
    
    # For each triplet i with weight w_i:
    # d(a_i)/d(pi) = 0.5 * sign(cross_i) * (-dy2_i, dx2_i)
    # d(a_i)/d(pj) = 0.5 * sign(cross_i) * (dy2_i, -dx2_i)
    # d(a_i)/d(pk) = 0.5 * sign(cross_i) * (-dy1_i, dx1_i)
    
    s = 0.5 * sign_cross * w  # shape (M,)
    
    # Accumulate gradients
    # d(soft_min)/d(pts) = -sum_i w_i * d(a_i)/d(pts)
    # So grad[pi] -= s_i * (-dy2_i, dx2_i) => grad[pi] += s_i * (dy2_i, -dx2_i)
    # Wait: d(soft_min)/d(pts) = -sum w_i * d(a_i)/d(pts)
    # d(a_i)/d(pi) = 0.5*sign*(-dy2, dx2)
    # So contribution to grad[pi] = -w_i * 0.5*sign*(-dy2, dx2) = w_i * 0.5*sign*(dy2, -dx2)
    
    # Let me redo this carefully.
    # soft_min = -1/beta * log(sum exp(-beta * a_i))
    # d(soft_min)/d(a_i) = -1/beta * (-beta * exp(-beta*a_i) / sum_exp) = exp(-beta*a_i)/sum_exp = w_i
    # d(soft_min)/d(pts) = sum_i w_i * d(a_i)/d(pts)
    # a_i = 0.5 * |cross_i|
    # d(a_i)/d(pt) = 0.5 * sign(cross_i) * d(cross_i)/d(pt)
    
    # So grad = sum_i w_i * 0.5 * sign(cross_i) * d(cross_i)/d(pts)
    
    # d(cross)/d(pi_x): cross = (pjx-pix)(pky-piy) - (p jy-piy)(pkx-pix)
    # = pjx*pky - pjx*piy - pix*pky + pix*piy - pjy*pkx + pjy*pix + piy*pkx - piy*pix
    # d/d(pix): -pky + piy + pjy - piy = pjy - pky = -dy2... wait
    # Let me just use the cross product formula directly.
    # cross = (pj - pi) x (pk - pi) in 2D = (pjx-pix)*(pky-piy) - (pjy-piy)*(pkx-pix)
    # d(cross)/d(pix) = -(pky-piy) + (pjy-piy) = -dy2 + dy1
    # Hmm, let me be more careful.
    # cross = dx1*dy2 - dy1*dx2
    # where dx1 = pjx-pix, dy1 = pjy-piy, dx2 = pkx-pix, dy2 = pky-piy
    # d(cross)/d(pix) = d/d(pix)[(pjx-pix)(pky-piy) - (pjy-piy)(pkx-pix)]
    # = -(pky-piy) - (pjy-piy)*(-1) = -dy2 + dy1
    # d(cross)/d(piy) = -(pjx-pix) + (pkx-pix) = -dx1 + dx2
    # d(cross)/d(pjx) = (pky-piy) = dy2
    # d(cross)/d(pjy) = -(pkx-pix) = -dx2
    # d(cross)/d(pkx) = -(pjy-piy) = -dy1
    # d(cross)/d(pky) = (pjx-pix) = dx1
    
    # So:
    # d(a_i)/d(pi) = 0.5*sign*(dy1-dy2, dx2-dx1)
    # d(a_i)/d(pj) = 0.5*sign*(dy2, -dx2)
    # d(a_i)/d(pk) = 0.5*sign*(-dy1, dx1)
    
    # grad[pi] += w_i * 0.5 * sign * (dy1_i - dy2_i, dx2_i - dx1_i)
    # grad[pj] += w_i * 0.5 * sign * (dy2_i, -dx2_i)
    # grad[pk] += w_i * 0.5 * sign * (-dy1_i, dx1_i)
    
    coeff = w * 0.5 * sign_cross  # shape (M,)
    
    # For pi (index _T_I):
    grad_x_pi = coeff * (dy1 - dy2)
    grad_y_pi = coeff * (dx2 - dx1)
    np.add.at(grad, (_T_I, 0), grad_x_pi)
    np.add.at(grad, (_T_I, 1), grad_y_pi)
    
    # For pj (index _T_J):
    grad_x_pj = coeff * dy2
    grad_y_pj = coeff * (-dx2)
    np.add.at(grad, (_T_J, 0), grad_x_pj)
    np.add.at(grad, (_T_J, 1), grad_y_pj)
    
    # For pk (index _T_K):
    grad_x_pk = coeff * (-dy1)
    grad_y_pk = coeff * dx1
    np.add.at(grad, (_T_K, 0), grad_x_pk)
    np.add.at(grad, (_T_K, 1), grad_y_pk)
    
    return soft_min, grad

def _cart_to_bary_grad(cart_grad):
    """Convert gradient from Cartesian to barycentric coordinates.
    cart_grad: shape (N, 2) gradient w.r.t. Cartesian coords.
    Returns gradient w.r.t. barycentric (u, v).
    """
    # x = 0.5 - 0.5*u  => dx/du = -0.5, dx/dv = 0
    # y = S3H - S3H*u  => dy/du = -S3H, dy/dv = 0
    # Wait, let me re-derive.
    # P = u*V0 + v*V1 + (1-u-v)*V2
    # x = u*0 + v*1 + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = u*0 + v*0 + (1-u-v)*S3H = S3H - S3H*u - S3H*v
    # dx/du = -0.5, dx/dv = 0.5
    # dy/du = -S3H, dy/dv = -S3H
    
    # grad_u = grad_x * dx/du + grad_y * dy/du
    # grad_v = grad_x * dx/dv + grad_y * dy/dv
    
    gu = cart_grad[:, 0] * (-0.5) + cart_grad[:, 1] * (-S3H)
    gv = cart_grad[:, 0] * (0.5) + cart_grad[:, 1] * (-S3H)
    return np.column_stack([gu, gv])

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
    max_starts = 60
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(30):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule for beta - more stages at high beta for better convergence
        betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(20):
                break
            
            # Define objective with analytic gradient in barycentric coords
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                val, cart_grad = _soft_min_area_and_grad_cart(cart, beta)
                bary_grad = _cart_to_bary_grad(cart_grad)
                return -val, -bary_grad.flatten()
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum (exact min)
        if _elapsed_ok(10):
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11})
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
                
                # Quick L-BFGS-B polish with gradient
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    val, cart_grad = _soft_min_area_and_grad_cart(cart, 500.0)
                    bary_grad = _cart_to_bary_grad(cart_grad)
                    return -val, -bary_grad.flatten()
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B', jac=True,
                                     options={'maxiter': 150, 'ftol': 1e-14, 'maxls': 30})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    if new_bary is not None:
                        current_bary = new_bary
                except Exception:
                    pass

    # Final polish on the absolute best found
    if best_bary is not None and _elapsed_ok(5):
        # First, try a high-beta soft-min polish with gradient for smoothness
        if _elapsed_ok(8):
            def final_soft_obj(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                val, cart_grad = _soft_min_area_and_grad_cart(cart, 50000.0)
                bary_grad = _cart_to_bary_grad(cart_grad)
                return -val, -bary_grad.flatten()
            
            try:
                res_soft = minimize(final_soft_obj, best_bary.flatten(), method='L-BFGS-B', jac=True,
                                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
                soft_bary = res_soft.x.reshape(N, 2)
                soft_bary = _project_bary_to_valid(soft_bary)
                _update_best(soft_bary)
            except Exception:
                pass
        
        # Then Nelder-Mead on the exact minimum
        if _elapsed_ok(3):
            def final_nm_obj(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13})
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
