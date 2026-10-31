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

def epigraph_penalty_obj(params, mu=100.0):
    """
    Epigraph formulation with penalty.
    params = [bary_flat (22), t]
    Minimize -t + mu * sum(max(0, t - area_i)^2)
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Violations: t - area_i where t > area_i
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    
    return -t + mu * penalty

def generate_symmetric_init():
    """
    Generate a symmetric configuration exploiting 3-fold symmetry.
    """
    bary = np.zeros((N, 2))
    
    # Near vertex 0: barycentric (1,0,0) -> (u=1,v=0)
    bary[0] = [0.95, 0.02]
    # Near vertex 1: barycentric (0,1,0) -> (u=0,v=1)
    bary[1] = [0.02, 0.95]
    # Near vertex 2: barycentric (0,0,1) -> (u=0,v=0)
    bary[2] = [0.02, 0.02]
    
    # Near edge midpoints (offset inward)
    bary[3] = [0.48, 0.03]  # edge 0-1
    bary[4] = [0.03, 0.48]  # edge 1-2
    bary[5] = [0.48, 0.48]  # edge 2-0
    
    # Center
    bary[6] = [1/3, 1/3]
    
    # Additional points: form a small hexagon around center
    r = 0.15
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        du = r * np.cos(angle)
        dv = r * np.sin(angle)
        bary[7 + i] = [1/3 + du, 1/3 + dv]
    
    # One more point
    bary[10] = [0.5, 0.25]
    
    return clamp_bary(bary)

def generate_radial_init():
    """
    Generate points using radial/angular parameterization from centroid.
    """
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3
    
    # Ring 1: 3 points at radius r1, angles 0, 120, 240
    r1 = 0.35
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[i] = [cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)]
    
    # Ring 2: 3 points at radius r2, angles 60, 180, 300
    r2 = 0.25
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 3
        bary[3 + i] = [cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)]
    
    # Ring 3: 3 points at radius r3, angles 0, 120, 240 (closer to center)
    r3 = 0.12
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[6 + i] = [cx + r3 * np.cos(angle), cy + r3 * np.sin(angle)]
    
    # Center point
    bary[9] = [cx, cy]
    
    # Additional point offset
    bary[10] = [0.5, 0.25]
    
    return clamp_bary(bary)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a range of scales to find a good starting point
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
    Refine a configuration using multiple optimization stages.
    Stage 1: Soft min (L-BFGS-B)
    Stage 2: Epigraph Penalty (SLSQP)
    Stage 3: Hard Min (Nelder-Mead)
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Epigraph refinement with SLSQP
    # Initial guess for t: current min area
    t_init = -hard_min_obj(bary_current.flatten())
    params_init = np.concatenate([bary_current.flatten(), [t_init]])
    bounds_ep = bounds_bary + [(0, 1)]
    
    try:
        result2 = minimize(
            epigraph_penalty_obj,
            params_init,
            args=(100.0,),
            method='SLSQP',
            bounds=bounds_ep,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        if result2.success:
            new_bary = clamp_bary(result2.x[:N*2].reshape(N, 2))
            # Check if it improved
            val_new = -hard_min_obj(new_bary.flatten())
            val_old = -hard_min_obj(bary_current.flatten())
            if val_new > val_old:
                bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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

    # 1. Try Symmetric Inits (High priority due to structure)
    sym_init = generate_symmetric_init()
    refined_sym = refine_local(sym_init, time_limit=8.0)
    update_best(refined_sym)
    
    rad_init = generate_radial_init()
    refined_rad = refine_local(rad_init, time_limit=8.0)
    update_best(refined_rad)
    
    # 2. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 3. Global Search with Differential Evolution
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
                maxiter=40,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 12 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(5.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more epigraph polish
            t_init = -hard_min_obj(best_bary.flatten())
            params_init = np.concatenate([best_bary.flatten(), [t_init]])
            bounds_ep = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_slsqp = minimize(
                epigraph_penalty_obj,
                params_init,
                args=(1000.0,), # Higher penalty for final polish
                method='SLSQP',
                bounds=bounds_ep,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if result_slsqp.success:
                new_bary = clamp_bary(result_slsqp.x[:N*2].reshape(N, 2))
                update_best(new_bary)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
