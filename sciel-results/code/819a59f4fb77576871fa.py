import numpy as np
from scipy.optimize import minimize
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
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def coordinated_ascent(bary_init, time_limit=3.0):
    """
    Targeted coordinated ascent: identify active (near-minimum) triangles 
    and move points to specifically increase those areas.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    while time.time() < start_time + time_limit:
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        # Identify active triangles (within 10% of minimum)
        active_threshold = min_area * 1.1
        active_mask = areas <= active_threshold
        active_tris = tri_idx[active_mask]
        
        if len(active_tris) == 0:
            break
            
        # For each active triangle, try to move one of its points to increase area
        # We use a simple gradient-free approach: try small perturbations
        improved = False
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        
        for _ in range(50):  # Try 50 random perturbations
            if time.time() > start_time + time_limit:
                break
                
            # Pick a random active triangle
            tri = active_tris[rng.integers(0, len(active_tris))]
            i, j, k = tri
            
            # Pick which point to move
            point_idx = [i, j, k][rng.integers(0, 3)]
            
            # Generate a random perturbation
            perturbation = rng.normal(0, 0.02, 2)
            new_bary = bary.copy()
            new_bary[point_idx] = new_bary[point_idx] + perturbation
            new_bary = clamp_bary(new_bary)
            
            new_areas = calc_areas_bary(new_bary)
            new_min_area = np.min(new_areas)
            
            if new_min_area > min_area:
                bary = new_bary
                min_area = new_min_area
                improved = True
        
        if not improved:
            # Try a larger perturbation
            perturbation = rng.normal(0, 0.05, (N, 2))
            new_bary = clamp_bary(bary + perturbation)
            new_areas = calc_areas_bary(new_bary)
            new_min_area = np.min(new_areas)
            
            if new_min_area > min_area:
                bary = new_bary
                min_area = new_min_area
            else:
                break
                
    return bary

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

    def track_good(bary):
        nonlocal best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        # Keep if within 2% of best or if we have fewer than 5 candidates
        if len(good_candidates) < 5 or val > best_val * 0.98:
            good_candidates.append((val, bary.copy()))
            # Keep only top 5
            good_candidates.sort(key=lambda x: -x[0])
            del good_candidates[5:]

    good_candidates = []

    # 1. Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    track_good(refined_hex)
    
    # 1b. Coordinated ascent on hex init
    if time.time() < deadline - 60:
        ca_hex = coordinated_ascent(hex_init, time_limit=5.0)
        refined_ca_hex = refine_local(ca_hex, time_limit=5.0)
        update_best(refined_ca_hex)
        track_good(refined_ca_hex)
    
    # 2. Random Init
    if time.time() < deadline - 50:
        rand_init = generate_random_init(seed=42)
        refined_rand = refine_local(rand_init, time_limit=6.0)
        update_best(refined_rand)
        track_good(refined_rand)
        
        # Coordinated ascent on random init
        if time.time() < deadline - 45:
            ca_rand = coordinated_ascent(rand_init, time_limit=4.0)
            refined_ca_rand = refine_local(ca_rand, time_limit=4.0)
            update_best(refined_ca_rand)
            track_good(refined_ca_rand)
    
    # 3. Basin Hopping
    if time.time() < deadline - 10 and best_bary is not None:
        current_bary = best_bary.copy()
        current_val = best_val
        rng_bh = np.random.default_rng(42)
        
        while time.time() < deadline - 5:
            # Perturb
            noise = rng_bh.normal(0, 0.2, current_bary.shape)
            new_bary = clamp_bary(current_bary + noise)
            
            # Local Refine
            refined = refine_local(new_bary, time_limit=min(3.0, deadline - 5 - time.time()))
            val = -hard_min_obj(refined.flatten())
            
            # Accept if better
            if val > current_val:
                current_val = val
                current_bary = refined
                update_best(refined)
                track_good(refined)
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
