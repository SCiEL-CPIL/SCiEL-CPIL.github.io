import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-7):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Define bounds for differential evolution
    # u in [0, 1], v in [0, 1] (clamped later)
    bounds = [(0, 1)] * (N * 2)
    
    best_bary = None
    best_val = -np.inf
    
    # 1. Run Differential Evolution
    # We use a relatively low maxiter to keep it fast, relying on polish later
    # popsize=15 is default, might be too slow. Let's try 10 or 15.
    # maxiter=50-100 should be enough for a good basin.
    
    try:
        de_result = differential_evolution(
            hard_min_obj,
            bounds,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-5,
            init='latinhypercube',
            polish=False, # We do our own polish
            disp=False
        )
        
        if de_result.success or de_result.fun < best_val:
            best_bary = clamp_bary(de_result.x.reshape(N, 2))
            best_val = -de_result.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        best_val = -hard_min_obj(best_bary.flatten())

    # 2. Local Polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            # Nelder-Mead is good for non-smooth final refinement
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            
            if new_val > best_val:
                best_bary = new_bary
                best_val = new_val
        except Exception as e:
            print(f"Polish failed: {e}")

    # 3. Convert to Cartesian and return
    cart_points = bary_to_cart(best_bary)
    return cart_points
