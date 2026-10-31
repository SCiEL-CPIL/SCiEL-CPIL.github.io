"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    5-stage approach: coarse -> medium -> sharp -> NM -> NM polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 4: Nelder-Mead for final polish on hard min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result4.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.9:
        return bary_current

    # Stage 5: Final Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result5.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Hexagonal lattice init
    hex_init = generate_hexagonal_init()
    initial_configs.append(hex_init)
    
    # Random inits with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initial_configs.append(generate_random_init(seed))
    
    # Perturbed hexagonal
    for sigma in [0.03, 0.05, 0.08]:
        initial_configs.append(generate_perturbed_init(hex_init, sigma, seed=42 + int(sigma*100)))
    
    # Refine each initial configuration
    for i, init_bary in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time()
        time_per_refine = min(remaining_time / max(1, len(initial_configs) - i), 15.0)
        
        refined = refine_local(init_bary, time_limit=time_per_refine)
        score = -hard_min_obj(refined.flatten())
        
        if score > best_score:
            best_score = score
            best_bary = refined.copy()
    
    # Try differential evolution if time remains
    if time.time() < deadline - 20:
        try:
            bounds = [(0, 1)] * (N * 2)
            de_result = differential_evolution(
                hard_min_obj,
                bounds=bounds,
                maxiter=30,
                popsize=20,
                tol=1e-6,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            de_score = -hard_min_obj(de_bary.flatten())
            
            if de_score > best_score:
                best_score = de_score
                best_bary = de_bary.copy()
        except Exception:
            pass
    
    # Final refinement of the best solution
    if time.time() < deadline - 5 and best_bary is not None:
        final_refined = refine_local(best_bary, time_limit=min(10.0, deadline - time.time() - 2))
        final_score = -hard_min_obj(final_refined.flatten())
        
        if final_score > best_score:
            best_score = final_score
            best_bary = final_refined.copy()
    
    # Convert to Cartesian coordinates
    if best_bary is None:
        # Fallback: simple valid configuration
        best_bary = np.column_stack([np.random.random(N), np.random.random(N) * (1 - np.random.random(N))])
        best_bary = clamp_bary(best_bary)
    
    cart_points = bary_to_cart(best_bary)
    
    return cart_points
