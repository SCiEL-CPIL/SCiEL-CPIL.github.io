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

def calc_min_area_excluding_point(bary, point_idx):
    """Calculate minimum triangle area excluding triangles containing point_idx."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Get all triangles that don't include point_idx
    mask = ~np.any(tri_idx == point_idx, axis=1)
    if not np.any(mask):
        return 1e10  # No triangles to consider
    
    tri_sub = tri_idx[mask]
    
    ui = u[tri_sub[:, 0]]
    vi = v[tri_sub[:, 0]]
    wi = w[tri_sub[:, 0]]
    
    uj = u[tri_sub[:, 1]]
    vj = v[tri_sub[:, 1]]
    wj = w[tri_sub[:, 1]]
    
    uk = u[tri_sub[:, 2]]
    vk = v[tri_sub[:, 2]]
    wk = w[tri_sub[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.min(np.abs(det))

def cyclic_coordinate_ascent(bary_init, time_limit=10.0, n_sweeps=50):
    """
    Perform cyclic coordinate ascent: optimize each point one at a time.
    For each point, find its optimal position given all others fixed.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    # Precompute which triangles include each point
    point_tri_masks = []
    for i in range(N):
        mask = np.any(tri_idx == i, axis=1)
        point_tri_masks.append(mask)
    
    for sweep in range(n_sweeps):
        if time.time() - start_time > time_limit:
            break
            
        for i in range(N):
            if time.time() - start_time > time_limit:
                    break
                    
            # Save current position
            orig_pos = bary[i].copy()
            
            # Optimize point i using Nelder-Mead in 2D
            def obj_2d(x):
                bary[i] = x
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            try:
                result = minimize(
                    obj_2d,
                    orig_pos,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                new_pos = result.x
                bary[i] = new_pos
                bary = clamp_bary(bary)
                
                new_val = -hard_min_obj(bary.flatten())
                if new_val > best_val:
                    best_val = new_val
            except Exception:
                bary[i] = orig_pos
    
    return bary

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

def generate_boundary_interior_init():
    """
    Generate initial configurations with points on triangle edges and interior.
    This exploits the geometric fact that boundary points maximize spread.
    """
    best_bary = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    # Strategy: place some points on edges, some in interior
    edge_configs = [
        # (num_on_edge0, num_on_edge1, num_on_edge2, num_interior)
        (3, 3, 3, 2),
        (4, 4, 2, 1),
        (3, 4, 3, 1),
        (5, 3, 2, 1),
        (4, 3, 3, 1),
    ]
    
    for n0, n1, n2, nint in edge_configs:
        if n0 + n1 + n2 + nint != N:
            continue
        
        for trial in range(20):
            bary = np.zeros((N, 2))
            
            # Edge 0: w=0, so u+v=1, v in [0,1]
            if n0 > 0:
                v_vals = rng.uniform(0.05, 0.95, n0)
                bary[:n0, 0] = 1 - v_vals
                bary[:n0, 1] = v_vals
            
            # Edge 1: v=0, so u+w=1, u in [0,1]
            if n1 > 0:
                start = n0
                u_vals = rng.uniform(0.05, 0.95, n1)
                bary[start:start+n1, 0] = u_vals
                bary[start:start+n1, 1] = 0.0
            
            # Edge 2: u=0, so v+w=1, v in [0,1]
            if n2 > 0:
                start = n0 + n1
                v_vals = rng.uniform(0.05, 0.95, n2)
                bary[start:start+n2, 0] = 0.0
                bary[start:start+n2, 1] = v_vals
            
            # Interior points
            if nint > 0:
                start = n0 + n1 + n2
                for k in range(nint):
                    u_val = rng.uniform(0.1, 0.8)
                    v_val = rng.uniform(0.1, 0.8 * (1 - u_val))
                    bary[start+k, 0] = u_val
                    bary[start+k, 1] = v_val
            
            bary = clamp_bary(bary)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        # Fallback to random
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using coordinated per-point ascent + global optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Coordinated per-point ascent (cyclic coordinate optimization)
    # This is the key structural change: optimize each point independently
    # while keeping others fixed, cycling through all points multiple times
    t1 = min(2.0, time_limit * 0.4)
    if time.time() - start_time < t1:
        bary_cyclic = cyclic_coordinate_ascent(bary_current, time_limit=t1, n_sweeps=30)
        val_cyclic = -hard_min_obj(bary_cyclic.flatten())
        if val_cyclic > best_val:
            best_val = val_cyclic
            bary_current = bary_cyclic
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: Smooth soft-min minimization for global coordination
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
            val_new = -hard_min_obj(bary_current.flatten())
            if val_new > best_val:
                best_val = val_new
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: More cyclic coordinate ascent with tighter convergence
    t3 = min(1.5, time_limit * 0.25)
    if time.time() - start_time < time_limit - 0.5:
        bary_cyclic2 = cyclic_coordinate_ascent(bary_current, time_limit=t3, n_sweeps=50)
        val_cyclic2 = -hard_min_obj(bary_cyclic2.flatten())
        if val_cyclic2 > best_val:
            best_val = val_cyclic2
            bary_current = bary_cyclic2
    
    # Stage 4: Final SLSQP polish
    if time.time() - start_time < time_limit - 0.2:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            bary_current = clamp_bary(result4.x.reshape(N, 2))
            val_final = -hard_min_obj(bary_current.flatten())
            if val_final > best_val:
                best_val = val_final
        except Exception:
            pass
        
    return bary_current

def basin_hopping(bary_init, time_limit=30.0):
    """
    Perform Basin Hopping on the soft-min objective.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    current_bary = bary_init.copy()
    current_val = -soft_min_obj(current_bary.flatten(), beta=50.0)
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    temperature = 0.05
    perturbation_scale = 0.1
    
    step = 0
    while time.time() - start_time < time_limit:
        # Local minimization
        refined = refine_local(current_bary, time_limit=min(2.0, time_limit - (time.time() - start_time)))
        refined_val = -soft_min_obj(refined.flatten(), beta=50.0)
        
        # Perturb
        noise = rng.normal(0, perturbation_scale, refined.shape)
        perturbed = clamp_bary(refined + noise)
        
        # Evaluate perturbed
        perturbed_val = -soft_min_obj(perturbed.flatten(), beta=50.0)
        
        # Metropolis acceptance
        delta = perturbed_val - refined_val
        if delta > 0 or np.log(rng.random()) < delta / temperature:
            current_bary = perturbed
            current_val = perturbed_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        else:
            pass
        
        # Update best from the local refinement itself
        if refined_val > best_val:
            best_val = refined_val
            best_bary = refined.copy()
            
        step += 1
        # Annealing
        temperature *= 0.95
        perturbation_scale *= 0.98
        
        # Check time
        if time.time() - start_time > time_limit * 0.9:
            break
            
    return best_bary

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

    # 1. Multiple diverse initializations with coordinated per-point refinement
    # Boundary-interior mixed initialization exploits geometric spread
    bi_init = generate_boundary_interior_init()
    refined_bi = refine_local(bi_init, time_limit=8.0)
    update_best(refined_bi)
    
    # Hexagonal initialization
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=5.0)
    update_best(refined_hex)
    
    # Random initialization with refinement
    rng = np.random.default_rng(123)
    for seed_trial in range(3):
        if time.time() > deadline - 60:
            break
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        rand_init = np.column_stack([u, v])
        refined_rand = refine_local(rand_init, time_limit=4.0)
        update_best(refined_rand)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 35:
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
                maxiter=15,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Basin Hopping on the best found so far
    if best_bary is not None and time.time() < deadline - 12:
        bh_time = min(20.0, deadline - start_time - 12.0)
        if bh_time > 5:
            bh_result = basin_hopping(best_bary, time_limit=bh_time)
            update_best(bh_result)
            
    # 4. Final High-Precision Polish with coordinated ascent
    if best_bary is not None and time.time() < deadline - 3:
        # Final cyclic coordinate ascent for precision
        final_time = min(5.0, deadline - start_time - 1.0)
        if final_time > 1:
            final_bary = cyclic_coordinate_ascent(best_bary, time_limit=final_time, n_sweeps=100)
            update_best(final_bary)
        
        # SLSQP polish
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
