import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def generate_symmetric_init():
    """
    Generate a highly symmetric initial configuration.
    1 center, 6 in a ring, 4 near vertices/edges.
    """
    pts = []
    center = [1/3, 1/3]
    pts.append(center)
    
    # Ring of 6
    r = 0.25
    for i in range(6):
        theta = 2 * np.pi * i / 6
        # Approximate circular motion in barycentric
        # Note: Barycentric "circle" is not a Euclidean circle, but this is a good heuristic
        du = r * np.cos(theta)
        dv = r * np.sin(theta) * 0.866
        u = center[0] + du
        v = center[1] + dv
        if u >= 0 and v >= 0 and u + v <= 1:
            pts.append([u, v])
        else:
            # Fallback to valid point
            pts.append([0.5, 0.25])

    # 4 more points: near vertices but inside
    eps = 0.1
    pts.append([eps, eps])
    pts.append([1-2*eps, eps])
    pts.append([eps, 1-2*eps])
    
    # One more to make 11? 
    # 1 + 6 + 3 = 10. Need 1 more.
    pts.append([0.5, 0.5]) # Midpoint of V1-V2 edge? No, that's on edge. 
    # Let's put it near the center of the top edge
    pts.append([0.25, 0.5])

    bary = np.array(pts[:N])
    return clamp_bary(bary)

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=2.0):
    """
    Refine a configuration using Nelder-Mead primarily, as it handles non-smooth mins well.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead
    try:
        result = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 2: SLSQP for final polish (handles bounds better)
    try:
        bounds = [(0, 1)] * (N * 2)
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-16}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def basin_hopping_step(bary, sigma, time_limit=1.0):
    """
    Perform a single basin hopping step: perturb, refine, accept if better or with probability.
    """
    rng = np.random.default_rng()
    noise = rng.normal(0, sigma, bary.shape)
    bary_perturbed = clamp_bary(bary + noise)
    
    # Refine the perturbed point
    bary_refined = refine_local(bary_perturbed, time_limit=time_limit)
    return bary_refined

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initial diverse starts
    inits = [
        generate_symmetric_init(),
        generate_random_init(seed=42),
        generate_random_init(seed=123),
        generate_random_init(seed=456)
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)
    
    # 2. Differential Evolution for broad exploration
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=40,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=5.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Basin Hopping / Iterated Local Search
    # This is the structural change: instead of just DE, we do explicit perturbation-refine loops
    rng = np.random.default_rng(1234)
    sigma = 0.1
    bh_iter = 0
    no_improve_count = 0
    
    while time.time() < deadline - 15:
        bh_iter += 1
        
        # Adaptive sigma: shrink if improving, grow if stuck
        if no_improve_count > 5:
            sigma *= 1.5
            no_improve_count = 0
        elif no_improve_count == 0:
            sigma *= 0.9
            
        # Ensure sigma is within reasonable bounds
        sigma = np.clip(sigma, 0.01, 0.3)
        
        # Perturb current best
        if best_bary is None:
            bary_start = generate_random_init(seed=1000 + bh_iter)
        else:
            noise = rng.normal(0, sigma, best_bary.shape)
            bary_start = clamp_bary(best_bary + noise)
            
        # Refine
        time_left = deadline - time.time()
        alloc_time = min(2.0, time_left * 0.1)
        if alloc_time < 0.5:
            break
            
        bary_candidate = refine_local(bary_start, time_limit=alloc_time)
        
        if update_best(bary_candidate):
            no_improve_count = 0
        else:
            no_improve_count += 1

    # 4. Final Polish
    if best_bary is not None and time.time() < deadline - 5:
        # High precision Nelder-Mead
        try:
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # SLSQP final
        if time.time() < deadline - 2:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-18}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
