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
AREA_TOTAL = np.sqrt(3) / 4.0

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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
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

def smooth_min_obj(bary_flat, beta):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this is equivalent to maximizing the soft-min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Handle zero areas to avoid log(0)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fallback to hard min if beta is too large causing overflow
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal-like lattice initialization.
    """
    pts = []
    
    # Row 0: 4 points along the bottom
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    
    # Row 1: 3 points offset
    for i in range(3):
        pts.append([(i + 0.5) / 3.0, H / 3.0])
    
    # Row 2: 2 points
    for i in range(2):
        pts.append([(i + 1.0) / 3.0, 2 * H / 3.0])
    
    # Row 3: 1 point at top
    pts.append([0.5, H])
    
    pts = np.array(pts)
    # Convert to barycentric
    # x = v + 0.5*w, y = w*H
    # w = y/H
    # v = x - 0.5*w
    # u = 1 - v - w
    bary_pts = []
    for p in pts:
        x, y = p
        w = y / H
        v = x - 0.5 * w
        u = 1 - v - w
        bary_pts.append([u, v])
        
    return np.array(bary_pts)

def generate_random_init(rng):
    """Generate random initial configuration in barycentric coords."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=10.0):
    """
    Refine a configuration using smooth annealing and hard polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth annealing
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [10, 50, 100, 200, 400]
    
    for beta in betas:
        if time.time() - start_time > time_limit * 0.7:
            break
            
        try:
            # Use L-BFGS-B for the smooth objective
            result = minimize(
                smooth_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            
            # Check if it improved
            curr_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            
            if new_val > curr_val:
                bary_current = new_bary
            else:
                # If no improvement, try SLSQP with current beta
                result2 = minimize(
                    smooth_min_obj,
                    bary_current.flatten(),
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                new_bary2 = clamp_bary(result2.x.reshape(N, 2))
                new_val2 = -hard_min_obj(new_bary2.flatten())
                if new_val2 > curr_val:
                    bary_current = new_bary2
                    
        except Exception:
            pass
        
    # Stage 2: Hard min polish with Nelder-Mead
    if time.time() - start_time < time_limit * 0.9:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            bary_current = clamp_bary(result_nm.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Final SLSQP polish on hard min
    if time.time() - start_time < time_limit:
        try:
            result_slsqp = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            bary_current = clamp_bary(result_slsqp.x.reshape(N, 2))
        except Exception:
            pass
        
    return bary_current

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

    # 1. Generate Initial Candidates
    candidates = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    candidates.append(hex_init)
    
    # Perturbed Hexagonal
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = hex_init + rng.normal(0, 0.05, size=hex_init.shape)
        candidates.append(clamp_bary(perturbed))
        
    # Random
    for _ in range(5):
        candidates.append(generate_random_init(rng))
        
    # 2. Global Search with Differential Evolution
    # Use smooth objective for DE to navigate better
    if time.time() < deadline - 60:
        try:
            def de_obj(x):
                return smooth_min_obj(x, beta=50.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            candidates.append(de_bary)
        except Exception:
            pass
            
    # 3. Local Refinement of Candidates
    n_cands = len(candidates)
    # Allocate time based on remaining time
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
            
        # Prioritize hexagonal and DE results
        if i < 4:
            t_limit = min(10.0, remaining)
        elif i < 6:
            t_limit = min(8.0, remaining)
        else:
            t_limit = min(5.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # 4. Second DE run if time permits
    if time.time() < deadline - 40:
        try:
            def de_obj2(x):
                return smooth_min_obj(x, beta=100.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de2 = differential_evolution(
                de_obj2,
                bounds,
                seed=123,
                maxiter=30,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            
            de_bary2 = clamp_bary(result_de2.x.reshape(N, 2))
            remaining = deadline - 20 - time.time()
            if remaining > 10:
                refined_de2 = refine_local(de_bary2, time_limit=min(15.0, remaining))
                update_best(refined_de2)
        except Exception:
            pass

    # 5. Final Polish on Best Solution
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            # High precision polish
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
            
    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
