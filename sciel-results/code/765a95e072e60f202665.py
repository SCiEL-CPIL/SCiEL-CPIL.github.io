import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

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

def energy_obj(bary_flat, t=0.01):
    """
    Energy-based objective: sum of inverse areas (repulsive potential).
    This is smooth and gradient-friendly.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid division by zero
    areas = np.maximum(areas, 1e-12)
    
    # Sum of 1/area^2 (stronger repulsion for small areas)
    # Or sum of -log(area)
    # Let's try a mix: -log(min) is non-smooth. 
    # Use: sum(1/area) is smooth but might not focus enough on min.
    # Use: -logsumexp(-areas/t) / N  -> approximates -min(areas)
    
    scaled = -areas / t
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / N  # This is actually -logsumexp(-A/t)/N ~ -min(A)/t? No.
    
    # Let's stick to a simpler energy: sum of 1/(area + eps)
    # This penalizes small areas heavily.
    return np.sum(1.0 / (areas + 1e-10))

def energy_grad(bary_flat, t=0.01):
    """
    Numerical gradient for the energy objective.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    areas = np.maximum(areas, 1e-12)
    
    # Objective: sum(1/area)
    # d(1/area)/d(bary) = -1/area^2 * d(area)/d(bary)
    
    # We need d(area)/d(bary) for each triangle.
    # Area = |det|. Since we want to maximize min, we assume det is positive for most.
    # Actually, let's just use numerical gradient for simplicity and robustness.
    
    eps = 1e-8
    grad = np.zeros_like(bary_flat)
    
    for i in range(len(bary_flat)):
        bary_plus = bary_flat.copy()
        bary_minus = bary_flat.copy()
        bary_plus[i] += eps
        bary_minus[i] -= eps
        
        val_plus = energy_obj(bary_plus, t)
        val_minus = energy_obj(bary_minus, t)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
        
    return grad

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
    Local refinement using gradient-based method (L-BFGS-B) on energy objective,
    followed by Nelder-Mead on hard min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Gradient-based refinement on smooth energy objective
    # This helps escape shallow local minima by using a smooth landscape
    try:
        result1 = minimize(
            energy_obj,
            bary_current.flatten(),
            jac=energy_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 20}
        )
        if result1.success or result1.fun < energy_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current

    # Stage 2: Nelder-Mead on hard min for precise optimization
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 3: Final polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def coordinate_descent_refine(bary_init, time_limit=5.0):
    """
    Cyclic coordinate descent: for each point, optimize its position while fixing others.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    best_bary = bary_current.copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            pt = bary_current[i].copy()
            
            areas_all = calc_areas_bary(bary_current)
            mask_involving_i = (tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | (tri_idx[:, 2] == i)
            areas_not_inv = areas_all[~mask_involving_i]
            min_area_not_inv = np.min(areas_not_inv) if len(areas_not_inv) > 0 else np.inf
            
            def obj_i(x):
                bary_temp = bary_current.copy()
                bary_temp[i, 0] = x[0]
                bary_temp[i, 1] = x[1]
                bary_temp = clamp_bary(bary_temp)
                
                areas_inv = calc_areas_bary(bary_temp)[mask_involving_i]
                min_inv = np.min(areas_inv)
                
                return -min(min_inv, min_area_not_inv)
            
            try:
                res = minimize(
                    obj_i,
                    pt,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                new_pt = res.x
                new_pt = np.clip(new_pt, 1e-9, 1-1e-9)
                if new_pt[0] + new_pt[1] > 1 - 1e-9:
                    scale = (1 - 1e-9) / (new_pt[0] + new_pt[1])
                    new_pt *= scale
                    new_pt = np.clip(new_pt, 1e-9, 1-1e-9)
                
                bary_current[i] = new_pt
                bary_current = clamp_bary(bary_current)
                
                new_val = -hard_min_obj(bary_current.flatten())
                if new_val > best_val + 1e-12:
                    best_val = new_val
                    best_bary = bary_current.copy()
                    improved = True
            except Exception:
                pass
                
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

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Hexagonal
    candidates.append(generate_hexagonal_init())
    
    # 2. Random starts
    for i in range(8):
        candidates.append(generate_random_init(seed=100 + i))
        
    # 3. Perturbed Hex
    hex_bary = candidates[0]
    for i in range(4):
        candidates.append(generate_perturbed_init(hex_bary, sigma=0.05, seed=200 + i))

    # Phase 2: Refine each candidate with gradient-based + NM
    refined_candidates = []
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15:
            break
            
        t_limit = min(4.0, deadline - time.time() - 5)
        if t_limit < 1.0:
            break
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            refined_candidates.append(refined)
            update_best(refined)
        except Exception:
            pass

    # Phase 3: Deep Refinement on Top Candidates
    def get_val(b):
        return -hard_min_obj(b.flatten())
        
    refined_candidates.sort(key=get_val, reverse=True)
    
    top_candidates = refined_candidates[:6]
    
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 8:
            break
        
        t_limit = min(10.0, deadline - time.time() - 3)
        if t_limit < 2.0:
            break
            
        try:
            # Coordinate descent
            cd_refined = coordinate_descent_refine(cand, time_limit=t_limit * 0.5)
            update_best(cd_refined)
            
            # Final local refine
            remaining = deadline - time.time() - 2
            if remaining > 2.0:
                final_refined = refine_local(cd_refined, time_limit=remaining)
                update_best(final_refined)
        except Exception:
            pass

    # Phase 4: Final Polish
    if best_bary is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()

    cart = bary_to_cart(best_bary)
    return cart
