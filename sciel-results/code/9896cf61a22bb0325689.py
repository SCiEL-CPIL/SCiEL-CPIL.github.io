import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Vectorized area computation using cross-product in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # For each triangle (i,j,k), area = |det([ui,vi,wi; uj,vj,wj; uk,vk,wk])|
    # Since ui+vi+wi=1, this equals |ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)|
    
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
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def rotate_bary_120(bary):
    """Rotate barycentric coordinates by 120 degrees: (u,v,w) -> (w,u,v)"""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    return np.column_stack([w, u])

def rotate_bary_240(bary):
    """Rotate by 240 degrees: (u,v,w) -> (v,w,u)"""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    return np.column_stack([v, w])

def build_symmetric_bary(params):
    """
    Build 11 barycentric points from symmetry parameters.
    
    Structure: 1 center + 3 orbits of 3 points (each orbit has 3 points related by 120° rotation)
    
    params layout:
    - params[0:2]: center point (u, v)
    - params[2:4]: orbit 1 representative (u, v)
    - params[4:6]: orbit 2 representative (u, v)
    - params[6:8]: orbit 3 representative (u, v)
    
    Total: 8 parameters -> 11 points
    """
    center = params[0:2]
    orbit1_rep = params[2:4]
    orbit2_rep = params[4:6]
    orbit3_rep = params[6:8]
    
    # Build orbits
    bary_list = [center]
    
    for rep in [orbit1_rep, orbit2_rep, orbit3_rep]:
        bary_list.append(rep)
        bary_list.append(rotate_bary_120(np.array([rep])))
        bary_list.append(rotate_bary_240(np.array([rep])))
    
    bary = np.array(bary_list)
    return clamp_bary(bary)

def build_general_bary(params):
    """
    Build 11 barycentric points from general parameters (no symmetry constraint).
    params: 22 values -> 11 points
    """
    bary = params.reshape(N, 2)
    return clamp_bary(bary)

def hard_min_obj_bary(bary):
    """Negative minimum area for hard minimization."""
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def epigraph_obj(params):
    """
    Epigraph formulation: maximize t subject to all areas >= t.
    We minimize -t.
    
    params: [bary_params..., t]
    For symmetric: 8 bary params + 1 t = 9 params
    """
    # Determine if symmetric or general based on length
    if len(params) == 9:
        bary = build_symmetric_bary(params[:8])
        t = params[8]
    else:
        bary = build_general_bary(params[:22])
        t = params[22]
    
    areas = calc_areas_bary(bary)
    
    # Penalty if any area < t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 1000.0 * penalty

def epigraph_obj_sym(params):
    """Epigraph with symmetric parameterization (9 params)."""
    bary = build_symmetric_bary(params[:8])
    t = params[8]
    
    areas = calc_areas_bary(bary)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 1000.0 * penalty

def epigraph_obj_general(params):
    """Epigraph with general parameterization (23 params)."""
    bary = build_general_bary(params[:22])
    t = params[22]
    
    areas = calc_areas_bary(bary)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 1000.0 * penalty

def soft_min_obj_sym(params, beta=50.0):
    """Soft-min surrogate for symmetric parameterization."""
    bary = build_symmetric_bary(params[:8])
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

def soft_min_obj_general(params, beta=50.0):
    """Soft-min surrogate for general parameterization."""
    bary = build_general_bary(params[:22])
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

def generate_symmetric_init(seed=None):
    """Generate a symmetric initial configuration."""
    rng = np.random.default_rng(seed)
    
    # Center near centroid
    center = np.array([1/3, 1/3]) + rng.normal(0, 0.02, 2)
    
    # Orbits: place representatives in different regions
    # Orbit 1: near one vertex
    orbit1_rep = np.array([0.5, 0.25]) + rng.normal(0, 0.05, 2)
    # Orbit 2: near another vertex  
    orbit2_rep = np.array([0.25, 0.5]) + rng.normal(0, 0.05, 2)
    # Orbit 3: near third vertex
    orbit3_rep = np.array([0.05, 0.05]) + rng.normal(0, 0.05, 2)
    
    params = np.concatenate([center, orbit1_rep, orbit2_rep, orbit3_rep])
    return params

def generate_hexagonal_symmetric_init():
    """Generate hexagonal lattice points, then fit to symmetric structure."""
    best_params = None
    best_val = -np.inf
    
    for scale in [0.18, 0.2, 0.22, 0.25, 0.28, 0.3]:
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
            # Try to select 11 points and fit to symmetric structure
            rng = np.random.default_rng(42)
            for trial in range(20):
                idx = rng.choice(len(pts), N, replace=False)
                selected = np.array(pts)[idx]
                
                # Find best symmetric fit: center = mean, orbits from remaining
                center_est = np.mean(selected, axis=0)
                center_est = np.clip(center_est, 0.01, 0.99)
                
                # Assign remaining 10 points to 3 orbits (3+3+3+1 leftover, or 4+3+3)
                # Simple approach: cluster by angle from center
                remaining = selected.copy()
                remaining[0] = center_est  # Mark center
                
                # Use 3 orbits of 3: pick 9 points, assign to orbits
                orbit_pts = []
                for orbit_i in range(3):
                    # Pick 3 points roughly 120° apart from center
                    angles = rng.uniform(0, 2*np.pi, 3)
                    orbit_rep = center_est + 0.2 * np.array([np.cos(angles[0]), np.sin(angles[0]) * 0.577])
                    orbit_rep = np.clip(orbit_rep, 0.01, 0.99)
                    orbit_pts.append(orbit_rep)
                
                params = np.concatenate([center_est, orbit_pts[0], orbit_pts[1], orbit_pts[2]])
                bary = build_symmetric_bary(params)
                val = -hard_min_obj_bary(bary)
                
                if val > best_val:
                    best_val = val
                    best_params = params.copy()
    
    if best_params is None:
        return generate_symmetric_init(42)
        
    return best_params

def refine_symmetric(params_init, time_limit=10.0):
    """Refine symmetric configuration using epigraph + local methods."""
    start_time = time.time()
    
    params_current = params_init.copy()
    
    # Stage 1: Epigraph with L-BFGS-B
    bounds_sym = [(0, 1)] * 8 + [(0, 0.1)]
    
    try:
        result1 = minimize(
            epigraph_obj_sym,
            params_current,
            method='L-BFGS-B',
            bounds=bounds_sym,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.fun < epigraph_obj_sym(params_current):
            params_current = result1.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        bary = build_symmetric_bary(params_current[:8])
        return bary
    
    # Stage 2: Soft-min refinement
    for beta in [30.0, 80.0, 200.0]:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result2 = minimize(
                soft_min_obj_sym,
                params_current[:8],
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_sym[:8],
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result2.success or result2.fun < soft_min_obj_sym(params_current[:8], beta):
                params_current[:8] = result2.x
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.8:
        bary = build_symmetric_bary(params_current[:8])
        return bary
    
    # Stage 3: Nelder-Mead on symmetric params
    try:
        result3 = minimize(
            lambda p: hard_min_obj_bary(build_symmetric_bary(p)),
            params_current[:8],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        params_current[:8] = result3.x
    except Exception:
        pass
    
    bary = build_symmetric_bary(params_current[:8])
    return bary

def refine_general(bary_init, time_limit=10.0):
    """Refine general (non-symmetric) configuration."""
    start_time = time.time()
    
    params = np.concatenate([bary_init.flatten(), [0.001]])
    bounds_general = [(0, 1)] * 22 + [(0, 0.1)]
    
    # Stage 1: Epigraph
    try:
        result1 = minimize(
            epigraph_obj_general,
            params,
            method='L-BFGS-B',
            bounds=bounds_general,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.fun < epigraph_obj_general(params):
            params = result1.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return build_general_bary(params[:22])
    
    # Stage 2: Soft-min
    for beta in [30.0, 80.0]:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result2 = minimize(
                soft_min_obj_general,
                params[:22],
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_general[:22],
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result2.success or result2.fun < soft_min_obj_general(params[:22], beta):
                params[:22] = result2.x
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.8:
        return build_general_bary(params[:22])
    
    # Stage 3: Nelder-Mead
    try:
        result3 = minimize(
            lambda p: hard_min_obj_bary(build_general_bary(p)),
            params[:22],
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        params[:22] = result3.x
    except Exception:
        pass
    
    return build_general_bary(params[:22])

def generate_general_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_bary(base_bary, sigma=0.03, seed=None):
    """Perturb a barycentric configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj_bary(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Symmetric initialization and refinement
    sym_init = generate_symmetric_init(42)
    sym_refined = refine_symmetric(sym_init, time_limit=12.0)
    update_best(sym_refined)
    
    # 2. Hexagonal symmetric init
    if time.time() < deadline - 30:
        hex_sym_init = generate_hexagonal_symmetric_init()
        hex_sym_refined = refine_symmetric(hex_sym_init, time_limit=10.0)
        update_best(hex_sym_refined)
    
    # 3. General random restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        remaining = deadline - 15 - time.time()
        if remaining < 4:
            break
        
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best
            init = generate_perturbed_bary(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif restart_count % 3 == 1:
            # Random
            init = generate_general_init(seed=3000 + restart_count)
        else:
            # Symmetric random
            sym_p = generate_symmetric_init(seed=4000 + restart_count)
            init = build_symmetric_bary(sym_p)
        
        t_limit = min(6.0, remaining * 0.7)
        if restart_count % 3 == 2:
            # Refine symmetric
            sym_p = init.flatten()[:8] if len(init.flatten()) >= 8 else generate_symmetric_init(seed=4000 + restart_count)
            refined = refine_symmetric(sym_p, time_limit=t_limit)
        else:
            refined = refine_general(init, time_limit=t_limit)
        update_best(refined)
        
        if restart_count > 20 and best_val > 0.005:
            break
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Try to extract symmetric structure from best and refine
        # Center estimate
        center_est = np.mean(best_bary, axis=0)
        
        # Try symmetric refinement with current best as guide
        sym_params = np.array([center_est[0], center_est[1]])
        
        # Assign points to orbits
        remaining_pts = best_bary.copy()
        # Remove center (closest to centroid)
        dists = np.linalg.norm(remaining_pts - center_est, axis=1)
        center_idx = np.argmin(dists)
        remaining_pts = np.delete(remaining_pts, center_idx, axis=0)
        
        # Cluster remaining 10 into 3 groups of 3 (one leftover)
        # Simple: sort by angle from center
        angles = np.arctan2(remaining_pts[:, 1] - center_est[1], remaining_pts[:, 0] - center_est[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = remaining_pts[sorted_idx]
        
        # Take 9 points (drop one), split into 3 orbits
        orbit1_rep = sorted_pts[0]
        orbit2_rep = sorted_pts[3]
        orbit3_rep = sorted_pts[6]
        
        sym_params = np.concatenate([center_est, orbit1_rep, orbit2_rep, orbit3_rep])
        
        sym_refined_final = refine_symmetric(sym_params, time_limit=min(8.0, deadline - 5 - time.time()))
        update_best(sym_refined_final)
        
        # Also try general refinement
        if time.time() < deadline - 4:
            general_refined = refine_general(best_bary, time_limit=min(5.0, deadline - 4 - time.time()))
            update_best(general_refined)
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
