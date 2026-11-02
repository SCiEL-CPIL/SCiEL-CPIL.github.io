"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y) in equilateral triangle."""
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Point = u*A + v*B + w*C where w = 1-u-v
    # x = 0*u + 1*v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = 0*u + 0*v + sqrt(3)/2 * w = sqrt(3)/2 * (1-u-v)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = (3.0 ** 0.5) / 2.0 * w
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # From above: y = sqrt(3)/2 * (1-u-v) => 1-u-v = 2*y/sqrt(3) => u+v = 1 - 2*y/sqrt(3)
    # x = v + 0.5*(1-u-v) = v + 0.5*(2*y/sqrt(3)) => v = x - y/sqrt(3)
    # u = 1 - 2*y/sqrt(3) - v = 1 - 2*y/sqrt(3) - x + y/sqrt(3) = 1 - x - y/sqrt(3)
    v = x - y / (3.0 ** 0.5)
    u = 1.0 - x - y / (3.0 ** 0.5)
    return np.column_stack([u, v])


def _triangle_areas_cart(points):
    """Compute all triangle areas from N points in Cartesian coords."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def _soft_min_neg(areas, alpha):
    """Negative soft-min of areas. Higher alpha = softer (more global)."""
    # soft_min = (1/alpha) * log(sum(exp(-alpha * a_i)))
    # We want to maximize soft_min, so minimize -soft_min
    # Use log-sum-exp for numerical stability
    neg_alpha_areas = -alpha * areas
    max_val = np.max(neg_alpha_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_alpha_areas - max_val)))
    soft_min = log_sum / alpha
    return -soft_min


def _objective(bary_params, alpha):
    """Objective function: negative soft-min of triangle areas."""
    bary = bary_params.reshape(-1, 2)
    # Clip to valid barycentric range with small margin
    bary = np.clip(bary, 0.0, 1.0)
    # Ensure u + v <= 1
    excess = np.maximum(0, bary[:, 0] + bary[:, 1] - 1.0)
    # Redistribute: scale down u and v proportionally
    mask = excess > 0
    if np.any(mask):
        scale = 1.0 / (bary[mask, 0] + bary[mask, 1])
        bary[mask, 0] *= scale
        bary[mask, 1] *= scale
    
    cart = _bary_to_cart(bary)
    areas = _triangle_areas_cart(cart)
    return _soft_min_neg(areas, alpha)


def _evaluate_score(cart_points):
    """Compute the actual score: minimum triangle area."""
    areas = _triangle_areas_cart(cart_points)
    return np.min(areas)


def _is_valid(cart_points):
    """Check if all points are inside the triangle."""
    x = cart_points[:, 0]
    y = cart_points[:, 1]
    # Inside triangle: y >= 0, y <= sqrt(3)/2 * x (left edge), y <= sqrt(3)/2 * (1-x) (right edge)
    # Actually for equilateral with vertices (0,0), (1,0), (0.5, sqrt(3)/2):
    # y >= 0
    # y <= sqrt(3) * x  (line from (0,0) to (0.5, sqrt(3)/2): slope = sqrt(3))
    # y <= sqrt(3) * (1 - x)  (line from (1,0) to (0.5, sqrt(3)/2): slope = -sqrt(3))
    eps = 1e-9
    return (np.all(y >= -eps) and 
            np.all(y <= (3.0 ** 0.5) * x + eps) and 
            np.all(y <= (3.0 ** 0.5) * (1 - x) + eps))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_score = 0.0
    best_cart = None
    
    # Initial random points as fallback
    rng = np.random.RandomState(42)
    
    # Generate initial barycentric points
    def _random_bary(n):
        u = rng.random(n)
        v = rng.random(n)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        return np.column_stack([u, v])
    
    # Multi-stage alpha schedule: start smooth, get sharper
    alpha_stages = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initial barycentric points
        bary_init = _random_bary(N)
        x0 = bary_init.flatten()
        
        # Multi-stage optimization
        current_x = x0
        for stage_idx, alpha in enumerate(alpha_stages):
            if time.time() > deadline:
                break
            
            # Try a few local optimizations at this alpha
            n_local = 3
            for local_idx in range(n_local):
                if time.time() > deadline:
                    break
                
                try:
                    result = minimize(
                        _objective,
                        current_x,
                        args=(alpha,),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.success or result.fun < _objective(current_x, alpha):
                        current_x = result.x
                except Exception:
                    pass
            
            # Evaluate actual score at this stage
            bary = current_x.reshape(-1, 2)
            bary = np.clip(bary, 0.0, 1.0)
            excess = np.maximum(0, bary[:, 0] + bary[:, 1] - 1.0)
            mask = excess > 0
            if np.any(mask):
                scale = 1.0 / (bary[mask, 0] + bary[mask, 1])
                bary[mask, 0] *= scale
                bary[mask, 1] *= scale
            cart = _bary_to_cart(bary)
            
            if _is_valid(cart):
                score = _evaluate_score(cart)
                if score > best_score:
                    best_score = score
                    best_cart = cart.copy()
        
        # Final high-precision polish with very sharp alpha
        if time.time() < deadline - 5.0:
            try:
                result = minimize(
                    _objective,
                    current_x,
                    args=(5000.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result.fun < _objective(current_x, 5000.0):
                    current_x = result.x
                    bary = current_x.reshape(-1, 2)
                    bary = np.clip(bary, 0.0, 1.0)
                    excess = np.maximum(0, bary[:, 0] + bary[:, 1] - 1.0)
                    mask = excess > 0
                    if np.any(mask):
                        scale = 1.0 / (bary[mask, 0] + bary[mask, 1])
                        bary[mask, 0] *= scale
                        bary[mask, 1] *= scale
                    cart = _bary_to_cart(bary)
                    if _is_valid(cart):
                        score = _evaluate_score(cart)
                        if score > best_score:
                            best_score = score
                            best_cart = cart.copy()
            except Exception:
                pass
    
    # Fallback if no valid points found
    if best_cart is None:
        bary = _random_bary(N)
        best_cart = _bary_to_cart(bary)
    
    return best_cart
# EVOLVE-BLOCK-END
