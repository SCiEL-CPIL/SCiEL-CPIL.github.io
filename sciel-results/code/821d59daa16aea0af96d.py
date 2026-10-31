import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    v = bary[:, 1]. copy()
    
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

def generate_symmetric_init():
    """
    Generate a configuration with 3-fold rotational symmetry.
    11 points: 1 at centroid, 3 orbits of 3 points.
    Total 1 + 9 = 10. We need 11.
    So: 1 at centroid, 1 on axis, 3 orbits of 3? Or 2 on axis, 3 orbits of 3?
    Let's try: 1 centroid, 3 orbits of 3, plus 1 extra on symmetry axis.
    """
    centroid = np.array([1.0/3, 1.0/3])
    
    def rotate_120(bary_pt):
        u, v = bary_pt
        w = 1 - u - v
        return np.array([w, u])
    
    def rotate_240(bary_pt):
        u, v = bary_pt
        w = 1 - u - v
        return np.array([v, w])
        
    best_bary = None
    best_val = -np.inf
    
    # Try varying radii for the 3 orbits
    radii_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    for r1 in radii_options:
        for r2 in radii_options:
            for r3 in radii_options:
                if r1 >= r2 or r2 >= r3: continue
                
                pts = [centroid.copy()]
                
                # Orbit 1
                ref1 = np.array([1/3 + r1, 1/3 - r1/2])
                ref1 = clamp_bary(ref1.reshape(1,2))[0]
                pts.extend([ref1, rotate_120(ref1), rotate_240(ref1)])
                
                # Orbit 2
                ref2 = np.array([1/3 + r2, 1/3 - r2/2])
                ref2 = clamp_bary(ref2.reshape(1,2))[0]
                pts.extend([ref2, rotate_120(ref2), rotate_240(ref2)])
                
                # Orbit 3
                ref3 = np.array([1/3 + r3, 1/3 - r3/2])
                ref3 = clamp_bary(ref3.reshape(1,2))[0]
                pts.extend([ref3, rotate_120(ref3), rotate_240(ref3)])
                
                # We have 10 points. Add 1 more on the vertical axis of symmetry (u=1/3)
                # Try different positions for the 11th point
                for extra_y in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    extra = np.array([1/3, 1/3 + extra_y])
                    extra = clamp_bary(extra.reshape(1,2))[0]
                    all_pts = np.array(pts + [extra])
                    val = -hard_min_obj(all_pts.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = all_pts.copy()
                        
    if best_bary is None:
        return generate_hexagonal_init()
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
    Robust local refinement using a mix of gradient (soft-min) and derivative-free (NM) methods.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(10.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min
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
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 4: Nelder-Mead on Hard Min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final Polish NM
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

def coordinate_descent_refine(bary_init, time_limit=5.0):
    """
    Cyclic coordinate descent: for each point, optimize its position while fixing others.
    This is often very effective for packing problems.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    best_bary = bary_current.copy()
    
    # We will iterate through points, and for each, try to move it to improve the min area.
    # We use a small Nelder-Mead search on just the 2 coordinates of the point.
    
    # Pre-calculate which triangles involve which point to speed up objective?
    # For simplicity, we just re-calculate all areas. N=11 is small.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            # Extract current point
            pt = bary_current[i].copy()
            
            # Objective for this point: minimize -min_area of triangles involving i
            # But we must consider ALL triangles? No, only those involving i change.
            # However, the min area of the *whole set* might be a triangle NOT involving i.
            # So we must minimize -min(min_area_involving_i, min_area_not_involving_i)
            
            # Pre-calc min area not involving i
            areas_all = calc_areas_bary(bary_current)
            # Identify triangles involving i
            # tri_idx is (165, 3). 
            # Mask where i is in any of the 3 positions
            mask_involving_i = (tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | (tri_idx[:, 2] == i)
            areas_not_inv = areas_all[~mask_involving_i]
            min_area_not_inv = np.min(areas_not_inv) if len(areas_not_inv) > 0 else np.inf
            
            def obj_i(x):
                # x is [u, v] for point i
                bary_temp = bary_current.copy()
                bary_temp[i, 0] = x[0]
                bary_temp[i, 1] = x[1]
                bary_temp = clamp_bary(bary_temp)
                
                areas_inv = calc_areas_bary(bary_temp)[mask_involving_i]
                min_inv = np.min(areas_inv)
                
                # The global min is min(min_inv, min_area_not_inv)
                # We want to maximize this. So minimize -min(...)
                return -min(min_inv, min_area_not_inv)
            
            try:
                res = minimize(
                    obj_i,
                    pt,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                new_pt = res.x
                # Ensure validity
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
    
    # 2. Symmetric
    candidates.append(generate_symmetric_init())
    
    # 3. Random starts
    for i in range(5):
        candidates.append(generate_random_init(seed=100 + i))
        
    # 4. Perturbed Hex
    hex_bary = candidates[0]
    for i in range(3):
        candidates.append(generate_perturbed_init(hex_bary, sigma=0.05, seed=200 + i))

    # Phase 2: Refine each candidate
    # Allocate time based on remaining budget
    num_candidates = len(candidates)
    # We have ~145s. 
    # Let's give each candidate ~10s for refinement, but stop if we run out.
    # Actually, we should prioritize the best ones.
    # Strategy: Run a quick refine on all, then a deep refine on the best few.
    
    refined_candidates = []
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Quick refine: 3 seconds
        t_limit = min(3.0, deadline - time.time() - 5)
        if t_limit < 1.0:
            break
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            refined_candidates.append(refined)
            update_best(refined)
        except Exception:
            pass

    # Phase 3: Deep Refinement on Top Candidates
    # Sort refined candidates by value
    def get_val(b):
        return -hard_min_obj(b.flatten())
        
    refined_candidates.sort(key=get_val, reverse=True)
    
    # Take top 5 for deeper refinement
    top_candidates = refined_candidates[:5]
    
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        
        # Deep refine: Coordinate Descent + NM
        t_limit = min(15.0, deadline - time.time() - 2)
        if t_limit < 2.0:
            break
            
        try:
            # First, try coordinate descent
            cd_refined = coordinate_descent_refine(cand, time_limit=t_limit * 0.6)
            update_best(cd_refined)
            
            # Then, try standard local refine on the result of CD
            remaining = deadline - time.time() - 2
            if remaining > 2.0:
                final_refined = refine_local(cd_refined, time_limit=remaining)
                update_best(final_refined)
        except Exception:
            pass

    # Phase 4: Final Polish on the absolute best
    if best_bary is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        try:
            # One last high-precision NM
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
