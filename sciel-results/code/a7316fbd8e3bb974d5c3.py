import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

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

def cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v."""
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
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

def clamp_bary(bary, eps=1e-9):
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

def active_triangles(bary, threshold=0.0):
    """Find triangles with area near the minimum (active constraints)."""
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    if min_area <= 0:
        return np.where(areas <= min_area + 1e-10)[0]
    # Triangles within threshold fraction of min
    active = np.where(areas <= min_area * (1.0 + threshold))[0]
    return active

def hard_min_obj(bary_flat):
    """Objective: negative of min triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate: negative log-sum-exp of areas.
    Minimizing this encourages all areas to be large.
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

def generate_heilbronn_init():
    """
    Generate initial configurations inspired by known Heilbronn triangle results.
    For n=11, use structured arrangements with boundary points.
    """
    candidates = []
    
    # Config 1: 3 corners, 4 edge quarters, 4 interior
    cfg1 = np.array([
        [0.0, 0.0],   # corner A
        [1.0, 0.0],   # corner B
        [0.0, 1.0],   # corner C
        [0.25, 0.0],  # edge AB quarter
        [0.75, 0.0],  # edge AB three-quarter
        [0.0, 0.25],  # edge AC quarter
        [0.0, 0.75],  # edge AC three-quarter
        [0.25, 0.5],  # edge BC
        [0.5, 0.25],  # edge BC
        [1/3, 1/3],   # centroid
        [0.3, 0.3],   # interior
    ])
    candidates.append(clamp_bary(cfg1[:11]))
    
    # Config 2: 3 corners, 3 edge midpoints, 5 interior (symmetric)
    cfg2 = np.array([
        [0.0, 0.0],   # corner
        [1.0, 0.0],   # corner
        [0.0, 1.0],   # corner
        [0.5, 0.0],   # edge midpoint AB
        [0.0, 0.5],   # edge midpoint AC
        [0.25, 0.75], # edge midpoint BC (in bary: u+v=1, so (0.25, 0.75))
        [1/3, 1/3],   # centroid
        [0.2, 0.3],   # interior
        [0.5, 0.2],   # interior
        [0.3, 0.5],   # interior
        [0.4, 0.4],   # interior
    ])
    candidates.append(clamp_bary(cfg2))
    
    # Config 3: Hexagonal-inspired
    cfg3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    candidates.append(clamp_bary(cfg3))
    
    # Config 4: Dense boundary
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.1, 0.9],
        [0.9, 0.1],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    candidates.append(clamp_bary(cfg4))
    
    # Config 5: Perturbed hexagonal
    rng = np.random.default_rng(42)
    base_hex = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.5],
        [0.5, 0.25],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    for _ in range(3):
        perturbed = base_hex + rng.normal(0, 0.03, size=base_hex.shape)
        candidates.append(clamp_bary(perturbed))
    
    return candidates

def generate_random_init(rng, n_candidates=5):
    """Generate random initial configurations."""
    candidates = []
    for _ in range(n_candidates):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
    return candidates

def generate_perturbed_best(rng, base_bary, n_perturb=5, scales=None):
    """Generate perturbed versions of a good configuration."""
    if scales is None:
        scales = [0.02, 0.03, 0.05, 0.08, 0.12]
    candidates = []
    for i in range(n_perturb):
        scale = scales[i % len(scales)]
        perturbed = base_bary + rng.normal(0, scale, size=base_bary.shape)
        candidates.append(clamp_bary(perturbed))
    return candidates

def coordinated_ascent(bary_init, time_limit=5.0, max_passes=10):
    """
    Coordinated ascent: iteratively move each point to maximize the minimum area,
    focusing on active (near-minimum) triangles.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_min = get_min_area(bary)
    
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            # Find which triangles involve point i and are active
            areas = calc_areas_bary(bary)
            min_area = np.min(areas)
            active = np.where(areas <= min_area * 1.05)[0]
            
            # Triangles involving point i
            tri_with_i = np.where((tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | (tri_idx[:, 2] == i))[0]
            # Intersection: active triangles that involve i
            relevant = np.intersect1d(active, tri_with_i)
            
            # Objective: maximize min area considering only triangles involving i
            def point_obj(xi):
                bary_temp = bary.copy()
                bary_temp[i] = xi
                bary_temp = clamp_bary(bary_temp)
                areas_temp = calc_areas_bary(bary_temp)
                # Focus on triangles involving i, but also consider global min
                if len(relevant) > 0:
                    # Weight: min of relevant triangles, penalize if global min decreases
                    rel_min = np.min(areas_temp[relevant])
                    global_min = np.min(areas_temp)
                    # Combine: want both to be large
                    return -(0.7 * rel_min + 0.3 * global_min)
                else:
                    return -np.min(areas_temp)
            
            try:
                res_i = minimize(
                    point_obj,
                    bary[i],
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                new_bary = bary.copy()
                new_bary[i] = clamp_bary(res_i.x.reshape(1, 2))[0]
                new_min = get_min_area(new_bary)
                if new_min > current_min + 1e-12:
                    bary = new_bary
                    current_min = new_min
                    improved = True
            except Exception:
                pass
        
        if not improved and pass_num > 2:
            break
    
    return bary, current_min

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple stages:
    1. Soft-min L-BFGS-B
    2. Coordinated ascent (targeted)
    3. Nelder-Mead polish
    4. Final coordinated ascent
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = get_min_area(bary_current)
    
    # Stage 1: Smooth soft-min minimization with L-BFGS-B
    if time.time() - start_time < time_limit * 0.2:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = get_min_area(new_bary)
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Stage 2: Coordinated ascent
    if time.time() - start_time < time_limit * 0.6:
        try:
            bary_ca, val_ca = coordinated_ascent(
                bary_current,
                time_limit=min(time_limit * 0.3, time_limit * 0.6 - (time.time() - start_time)),
                max_passes=5
            )
            if val_ca > current_val:
                bary_current = bary_ca
                current_val = val_ca
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on full configuration
    if time.time() - start_time < time_limit * 0.8:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = get_min_area(new_bary)
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Stage 4: Final coordinated ascent
    if time.time() - start_time < time_limit * 0.95:
        try:
            remaining = time_limit * 0.95 - (time.time() - start_time)
            if remaining > 0.5:
                bary_ca2, val_ca2 = coordinated_ascent(
                    bary_current,
                    time_limit=remaining,
                    max_passes=3
                )
                if val_ca2 > current_val:
                    bary_current = bary_ca2
                    current_val = val_ca2
        except Exception:
            pass
    
    return bary_current

def de_objective(x_flat):
    """Objective for differential evolution."""
    bary = x_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    return -get_min_area(bary)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    candidates = []
    
    # 1. Heilbronn-inspired configurations
    candidates.extend(generate_heilbronn_init())
    
    # 2. Random candidates
    rng = np.random.default_rng(42)
    candidates.extend(generate_random_init(rng, n_candidates=5))

    # Process candidates with time allocation
    # Prioritize structured inits, then random
    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        remaining = deadline - 20 - time.time()
        if remaining < 2:
            break
            
        # Allocate more time to structured inits (first 8)
        if i < 8:
            t_limit = min(6.0, remaining)
        else:
            t_limit = min(3.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Iterative perturbation and refinement of best
    if best_bary is not None:
        rng2 = np.random.default_rng(999)
        for iter_num in range(8):
            if time.time() > deadline - 15:
                break
            
            remaining = deadline - 15 - time.time()
            if remaining < 3:
                break
            
            # Generate perturbations with decreasing scales
            scale = 0.15 * (0.7 ** iter_num)
            perturb_cands = generate_perturbed_best(
                rng2, best_bary, n_perturb=2, scales=[scale, scale * 0.5]
            )
            
            for pc in perturb_cands:
                if time.time() > deadline - 12:
                    break
                remaining2 = deadline - 12 - time.time()
                if remaining2 < 2:
                    break
                try:
                    refined_p = refine_local(pc, time_limit=min(4.0, remaining2))
                    update_best(refined_p)
                except Exception:
                    continue
    
    # Differential Evolution Global Search if time permits
    if time.time() < deadline - 30:
        try:
            # DE bounds in barycentric space [0, 1] for u and v
            # But we need u+v <= 1. The clamp function handles this.
            de_bounds = [(0, 1)] * (N * 2)
            
            # Use a smaller population and fewer iterations to fit in remaining time
            # N=11, 22 variables.
            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-5,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result is not None and de_result.x is not None:
                de_bary = clamp_bary(de_result.x.reshape(N, 2))
                de_val = get_min_area(de_bary)
                
                # If DE found something better, or if it's close, refine it
                if de_val > best_val * 0.95 or de_val > best_val:
                    # Refine the DE result
                    remaining_de = deadline - 15 - time.time()
                    if remaining_de > 5:
                        refined_de = refine_local(de_bary, time_limit=min(10, remaining_de))
                        update_best(refined_de)
                    else:
                        update_best(de_bary)
        except Exception:
            pass

    # Final polish on best solution
    if best_bary is not None and time.time() < deadline - 3:
        try:
            remaining = deadline - 3 - time.time()
            if remaining > 1:
                bary_final, val_final = coordinated_ascent(
                    best_bary,
                    time_limit=remaining,
                    max_passes=5
                )
                if val_final > best_val:
                    best_bary = bary_final
                    best_val = val_final
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_heilbronn_init()[0]

    cart = bary_to_cart(best_bary)
    
    return cart
