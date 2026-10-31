import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u >= 0, v >= 0, u + v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_value(bary):
    """Compute the hard minimum area value."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def smooth_obj(x, beta):
    """
    Smooth objective for optimization.
    Minimizes -soft_min(areas).
    soft_min(A) = -1/beta * log(sum(exp(-beta * A)))
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid numerical issues with very small areas
    areas = np.maximum(areas, 1e-12)
    
    # Calculate log-sum-exp for stability
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # If max_scaled is very large positive, exp will overflow. 
    # However, areas are positive, so -beta*areas is negative.
    # max_scaled will be the least negative (closest to 0).
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Bounds for barycentric coordinates u, v
    # u >= 0, v >= 0, u + v <= 1
    # We use bounds [0, 1] for both and rely on clamp_bary inside the objective,
    # but DE works better with explicit constraints if possible. 
    # Since DE in scipy handles bounds, we set bounds to [0, 1] for u and v.
    # The constraint u+v<=1 is handled by the projection/clamping inside the objective function.
    # Note: This is slightly inefficient as DE explores invalid regions, but clamp_bary
    # projects them back, effectively creating a constrained optimization over the triangle.
    
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Phase 1: Global Optimization using Differential Evolution
    # We use a moderate beta for the smooth objective to guide the global search.
    # A lower beta makes the landscape smoother and easier for DE to navigate.
    beta_global = 20.0
    
    try:
        # DE can be slow. We limit maxiter and popsize to fit within time.
        # N=11, 22 variables.
        de_result = differential_evolution(
            smooth_obj,
            bounds=bounds,
            args=(beta_global,),
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.2),
            recombination=0.9,
            seed=42,
            polish=False,  # We will do our own polish
            init='latinhypercube'
        )
        
        best_x = de_result.x
        best_val_de = -de_result.fun
        
    except Exception as e:
        # Fallback to random if DE fails
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_x = np.column_stack([u, v]).flatten()
        best_val_de = hard_min_value(best_x.reshape(N, 2))

    if time.time() > deadline - 20:
        # Time check: if we are close to deadline, skip to final conversion
        pass
    else:
        # Phase 2: Local Refinement with L-BFGS-B on High-Beta Smooth Objective
        # Increase beta to make the smooth objective closer to the hard min.
        beta_local = 100.0
        
        try:
            res_local = minimize(
                smooth_obj,
                best_x,
                args=(beta_local,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_local.success or res_local.fun < de_result.fun:
                best_x = res_local.x
        except Exception:
            pass

        # Phase 3: Final Polish with Nelder-Mead on Hard Min
        # This ensures we are exactly at the local optimum of the hard problem.
        if time.time() < deadline - 5:
            try:
                def hard_obj(x):
                    return -hard_min_value(x.reshape(N, 2))
                
                res_nm = minimize(
                    hard_obj,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
                )
                if -res_nm.fun > -de_result.fun:
                    best_x = res_nm.x
            except Exception:
                pass

    # Convert best barycentric coordinates to Cartesian
    best_bary = clamp_bary(best_x.reshape(N, 2))
    cart = bary_to_cart(best_bary)
    
    return cart
