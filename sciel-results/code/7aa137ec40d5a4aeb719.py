import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

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

def get_areas_cartesian(cart):
    """Compute all triangle areas."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    areas = get_areas_cartesian(points)
    return np.min(areas)

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

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

def epigraph_obj(z, beta=50.0, lambda_pen=10.0):
    """
    Smooth epigraph objective.
    z: [bary_1, bary_2, ..., bary_N, t]
    Minimize -t + lambda * sum(softplus(beta * (t - area_i)))
    """
    bary = z[:N*2].reshape(N, 2)
    t = z[-1]
    
    # Ensure bary is valid for area calculation
    bary_clamped = clamp_bary(bary)
    areas = calc_areas_bary(bary_clamped)
    
    # softplus(x) = log(1 + exp(x))
    diff = t - areas
    diff_clipped = np.clip(diff, -50, 50)
    sp = np.logaddexp(0, beta * diff_clipped)
    
    penalty = lambda_pen * np.sum(sp)
    obj = -t + penalty
    
    return obj

def generate_lattice_init():
    """Generate a triangular lattice initialization."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

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
                val = -hard_min_obj_bary(bary.flatten())
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

def refine_epigraph(bary_init, t_init=0.01, time_limit=5.0):
    """
    Refine using graduated epigraph optimization.
    """
    start_time = time.time()
    bary = bary_init.copy()
    t = t_init
    
    # Sequence of betas for gradual sharpening
    betas = [5.0, 20.0, 100.0, 500.0]
    lambdas = [10.0, 10.0, 5.0, 1.0]
    
    for idx, beta in enumerate(betas):
        if time.time() - start_time > time_limit - 1:
            break
            
        lambda_pen = lambdas[idx]
        
        # Initial guess for z: [bary_flat, t]
        z0 = np.concatenate([bary.flatten(), [t]])
        
        # Bounds: u in [0,1], v in [0,1], t in [0, 0.1]
        bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
        
        try:
            res = minimize(
                epigraph_obj,
                z0,
                method='SLSQP',
                bounds=bounds,
                args=(beta, lambda_pen),
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or (res.fun < epigraph_obj(z0, beta, lambda_pen)):
                z_opt = res.x
                bary = clamp_bary(z_opt[:N*2].reshape(N, 2))
                t = z_opt[-1]
        except Exception:
            pass
            
    return bary, t

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
        if result1.success or result1.fun < -hard_min_obj_bary(bary_current.flatten()):
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
        if result2.success or result2.fun < -hard_min_obj_bary(bary_current.flatten()):
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
        if result3.success or result3.fun < -hard_min_obj_bary(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Nelder-Mead on hard min for fine tuning
    try:
        result4 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result4.fun < -hard_min_obj_bary(bary_current.flatten()):
            bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result5.fun < -hard_min_obj_bary(bary_current.flatten()):
            bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary, val=None):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        if val is None:
            val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    bary_ref, t_ref = refine_epigraph(lat_init, t_init=0.005, time_limit=10.0)
    update_best(bary_ref)
    
    # 2. Hexagonal Init
    hex_init = generate_hexagonal_init()
    bary_ref2 = refine_local(hex_init, time_limit=10.0)
    update_best(bary_ref2)
    
    # 3. Differential Evolution for Global Search
    # DE works on barycentric coordinates directly
    # We need to map DE results to valid barycentric coords
    def de_objective(x_flat):
        """Objective for DE: x_flat is 2N values in [-1, 1] which we map to barycentric."""
        # Map from [-1, 1]^22 to valid barycentric simplex
        # Use a simple mapping: for each point, u = (x[2i]+1)/2, v = (x[2i+1]+1)/2 * (1-u)
        # This ensures u in [0,1], v in [0, 1-u]
        x = x_flat.reshape(N, 2)
        u = (x[:, 0] + 1) / 2.0
        v = (x[:, 1] + 1) / 2.0 * (1 - u)
        bary = np.column_stack([u, v])
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        min_area = calc_min_area_cartesian(cart)
        return -min_area
    
    # DE bounds: each coordinate in [-1, 1]
    de_bounds = [(-1, 1)] * (N * 2)
    
    if time.time() < deadline - 60:
        try:
            # Run DE with moderate population
            de_result = differential_evolution(
                de_objective,
                de_bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,  # We'll do our own polishing
                updating='deferred',
                workers=1
            )
            
            if de_result.success or de_result.fun < -best_val:
                # Map DE result back to barycentric
                x_de = de_result.x.reshape(N, 2)
                u = (x_de[:, 0] + 1) / 2.0
                v = (x_de[:, 1] + 1) / 2.0 * (1 - u)
                bary_de = np.column_stack([u, v])
                bary_de = clamp_bary(bary_de)
                
                # Refine the DE result locally
                bary_refined = refine_local(bary_de, time_limit=15.0)
                update_best(bary_refined)
                
        except Exception:
            pass
    
    # 4. Random Restarts with Local Refinement
    rng = np.random.default_rng(42)
    
    num_random_starts = 12
    
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
            
        init_bary = generate_random_init(seed=100 + i)
        bary_ref = refine_local(init_bary, time_limit=4.0)
        update_best(bary_ref)
        
        # Occasionally perturb the best
        if i % 3 == 0 and best_bary is not None and time.time() < deadline - 40:
            noise = rng.normal(0, 0.05, best_bary.shape)
            perturbed = clamp_bary(best_bary + noise)
            bary_ref = refine_local(perturbed, time_limit=4.0)
            update_best(bary_ref)

    # 5. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 5:
        # Use Nelder-Mead on the hard min for final tweak
        try:
            res_nm = minimize(
                hard_min_obj_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_val: # i.e., -min_area < -best_val => min_area > best_val
                update_best(clamp_bary(res_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more SLSQP with very high beta
        if time.time() < deadline - 2:
            try:
                t_init = best_val
                z0 = np.concatenate([best_bary.flatten(), [t_init]])
                bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
                res_final = minimize(
                    epigraph_obj,
                    z0,
                    method='SLSQP',
                    bounds=bounds,
                    args=(1000.0, 0.1),
                    options={'maxiter': 1000, 'ftol': 1e-14}
                )
                if res_final.fun < epigraph_obj(z0, 1000.0, 0.1):
                    bary_opt = clamp_bary(res_final.x[:N*2].reshape(N, 2))
                    update_best(bary_opt)
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
