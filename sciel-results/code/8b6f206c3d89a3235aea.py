import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

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

def epigraph_obj(x):
    """
    Epigraph formulation: x = [bary (N*2), t]
    Minimize t + penalty for violated constraints (areas < t)
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas below t
    violations = np.maximum(t - areas, 0)
    penalty = np.sum(violations ** 2)
    
    return t + 1000.0 * penalty

def epigraph_obj_with_t(x):
    """
    Epigraph with explicit t minimization.
    x = [bary (N*2), t]
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # We want to minimize t subject to all areas >= t
    # Use a smooth approximation: minimize t + sum(max(0, t - area_i)^2)
    violations = np.maximum(t - areas, 0)
    penalty = np.sum(violations ** 2)
    
    return t + 500.0 * penalty

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

def coordinate_descent(bary, n_iter=50, step_size=0.01, time_limit=None, start_time=None):
    """
    Coordinate descent: optimize one point at a time.
    For each point, try to move it to maximize the minimum area.
    """
    if start_time is None:
        start_time = time.time()
    
    bary = bary.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    for iteration in range(n_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            
            best_local = bary[i].copy()
            best_local_val = current_val
            
            # Try small perturbations in 8 directions
            for du in [-step_size, 0, step_size]:
                for dv in [-step_size, 0, step_size]:
                    if du == 0 and dv == 0:
                        continue
                    new_bary = bary.copy()
                    new_bary[i] = bary[i] + [du, dv]
                    new_bary = clamp_bary(new_bary)
                    val = -hard_min_obj(new_bary.flatten())
                    if val > best_local_val + 1e-12:
                        best_local_val = val
                        best_local = new_bary[i].copy()
            
            if best_local_val > current_val + 1e-12:
                bary[i] = best_local
                current_val = best_local_val
                improved = True
        
        if not improved:
            # Try smaller step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages including epigraph.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Epigraph formulation
    try:
        # Initialize t with current min area
        areas_init = calc_areas_bary(bary_current)
        t_init = np.min(areas_init) * 0.9
        
        x_init = np.concatenate([bary_current.flatten(), [t_init]])
        bounds_epigraph = bounds_bary + [(0, 1)]
        
        result1 = minimize(
            epigraph_obj_with_t,
            x_init,
            method='L-BFGS-B',
            bounds=bounds_epigraph,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result1.success or -result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x[:N*2].reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Soft-min refinement
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.75:
        return bary_current
    
    # Stage 4: Coordinate descent polish
    try:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0.5:
            bary_current = coordinate_descent(
                bary_current, 
                n_iter=30, 
                step_size=0.02,
                time_limit=remaining_time * 0.8,
                start_time=start_time
            )
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
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
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Epigraph polish
        try:
            areas_final = calc_areas_bary(best_bary)
            t_final = np.min(areas_final) * 0.95
            x_final = np.concatenate([best_bary.flatten(), [t_final]])
            bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_epigraph = minimize(
                epigraph_obj_with_t,
                x_final,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            update_best(clamp_bary(result_epigraph.x[:N*2].reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            # SLSQP polish
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 1.5:
            # Nelder-Mead final
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Coordinate descent final polish
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                remaining = deadline - 0.5 - time.time()
                if remaining > 0.3:
                    best_bary = coordinate_descent(
                        best_bary,
                        n_iter=20,
                        step_size=0.005,
                        time_limit=remaining,
                        start_time=start_time
                    )
                    update_best(best_bary)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
