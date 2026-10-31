import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def generate_hexagonal_template():
    """
    Generate a hexagonal lattice template in barycentric coordinates.
    This serves as the base structure for the deformable lattice.
    """
    # Hexagonal packing: points at regular intervals
    # In barycentric coords, this is a triangular lattice
    scale = 0.3
    pts = []
    
    # Row w=0 (bottom edge)
    for j in range(4):
        u = j * scale
        v = 0.0
        if u + v <= 1.0:
            pts.append([u, v])
    
    # Row w=scale
    w_val = scale
    for j in range(3):
        u = j * scale
        v = (1 - w_val) - u
        if u >= -1e-9 and v >= -1e-9:
            pts.append([u, v])
    
    # Row w=2*scale
    w_val = 2 * scale
    for j in range(2):
        u = j * scale
        v = (1 - w_val) - u
        if u >= -1e-9 and v >= -1e-9:
            pts.append([u, v])
    
    # Row w=3*scale
    w_val = 3 * scale
    u = 0.0
    v = (1 - w_val) - u
    if u >= -1e-9 and v >= -1e-9:
        pts.append([u, v])
    
    if len(pts) >= N:
        return np.array(pts[:N])
    elif len(pts) > 0:
        # Pad with center if needed
        while len(pts) < N:
            pts.append([1/3, 1/3])
        return np.array(pts[:N])
    else:
        # Fallback: random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return np.column_stack([u, v])

def deformable_lattice_params_to_bary(params):
    """
    Convert global deformation parameters to barycentric coordinates.
    
    params layout (8 + 22 = 30 values):
    [0]: global_scale
    [1]: shear_x (horizontal shear)
    [2]: shear_y (vertical shear)  
    [3]: rotation angle (small)
    [4]: offset_u (global shift in u)
    [5]: offset_v (global shift in v)
    [6]: u_amplitude (per-point u offset scale)
    [7]: v_amplitude (per-point v offset scale)
    [8:30]: per-point offsets (11 points * 2 = 22 values)
    """
    global_scale = params[0]
    shear_x = params[1]
    shear_y = params[2]
    rotation = params[3]
    offset_u = params[4]
    offset_v = params[5]
    u_amp = params[6]
    v_amp = params[7]
    
    template = generate_hexagonal_template()
    
    # Apply affine transformation in barycentric space
    # Barycentric affine: [u, v] -> M @ [u, v] + t
    # Where M is 2x2, t is 2-vector
    
    # Build transformation matrix
    c, s = np.cos(rotation), np.sin(rotation)
    M = np.array([
        [global_scale * c + shear_x, -global_scale * s],
        [global_scale * s + shear_y, global_scale * c]
    ])
    t = np.array([offset_u, offset_v])
    
    # Apply to template
    bary = template @ M.T + t
    
    # Add per-point offsets
    offsets = params[8:30].reshape(N, 2)
    bary[:, 0] += u_amp * offsets[:, 0]
    bary[:, 1] += v_amp * offsets[:, 1]
    
    return bary

def deformable_obj(params):
    """
    Objective function for the deformable lattice parameterization.
    """
    bary = deformable_lattice_params_to_bary(params)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Use soft min for smoothness
    if np.any(areas < 1e-15):
        return 1e10
    
    beta = 80.0
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_random_init():
    """Generate a random initial configuration."""
    rng = np.random.default_rng(42)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, scale=0.05):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(123)
    noise = rng.normal(0, scale, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
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

    # 1. Start with hexagonal template
    template_bary = generate_hexagonal_template()
    update_best(template_bary)
    
    # 2. Global search using Differential Evolution on deformable parameters
    #    This explores the 30-dimensional parameter space of the lattice
    de_start = time.time()
    de_time_budget = min(45.0, deadline - start_time - 30.0)
    
    if de_time_budget > 10:
        try:
            # Define bounds for each parameter
            de_bounds = [
                (0.5, 1.5),      # global_scale
                (-0.2, 0.2),     # shear_x
                (-0.2, 0.2),     # shear_y
                (-0.3, 0.3),     # rotation
                (-0.15, 0.15),   # offset_u
                (-0.15, 0.15),   # offset_v
                (0.0, 0.1),      # u_amplitude
                (0.0, 0.1),      # v_amplitude
            ] + [(-0.1, 0.1)] * 22  # per-point offsets
            
            def de_obj_wrapper(params):
                if time.time() > deadline - 5:
                    return 1e10
                return deformable_obj(params)
            
            result_de = differential_evolution(
                de_obj_wrapper,
                de_bounds,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                update_interval=1,
                workers=-1
            )
            
            if result_de.success or result_de.fun < 1e9:
                de_bary = deformable_lattice_params_to_bary(result_de.x)
                update_best(de_bary)
        except Exception:
            pass
    
    # 3. Local refinement on the best found
    if best_bary is not None:
        refined_best = refine_local(best_bary, time_limit=min(15.0, deadline - start_time - 10.0))
        update_best(refined_best)
    
    # 4. Multiple restarts with different random seeds
    if best_bary is not None:
        remaining = deadline - start_time - 5.0
        if remaining > 10:
            rng = np.random.default_rng(999)
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Perturb the best solution
                perturbed = generate_perturbed_init(best_bary, scale=0.03)
                refined = refine_local(perturbed, time_limit=min(5.0, deadline - start_time - 5.0))
                update_best(refined)
    
    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
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
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
