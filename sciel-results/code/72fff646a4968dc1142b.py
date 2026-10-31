import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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

def single_point_obj(point_flat, fixed_bary, point_idx, beta=100.0):
    """Objective for moving a single point while others are fixed."""
    bary = fixed_bary.copy()
    bary[point_idx] = point_flat
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-15):
        return 1e10
    if beta < 1e6:
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
        return -soft_min
    else:
        return -np.min(areas)

def sequential_relocation(bary_init, time_limit=5.0, n_passes=3):
    """
    Structural change: coordinate descent - optimize one point at a time.
    Each sub-problem is 2D, much simpler than the full 22D problem.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    for p in range(n_passes):
        for i in range(N):
            if time.time() > start_time + time_limit:
                return bary
            fixed = bary.copy()
            pt = bary[i].flatten()
            bounds = [(0, 1), (0, 1)]
            
            # Multi-start for single point: try a few perturbations
            best_pt = pt.copy()
            best_pt_val = -hard_min_obj(bary.flatten())
            
            rng = np.random.default_rng(hash((i, p)) % 100000)
            starts = [pt]
            for _ in range(2):
                noise = rng.normal(0, 0.05, 2)
                starts.append(pt + noise)
            
            for start_pt in starts:
                start_pt = np.clip(start_pt, 1e-9, 1 - 1e-9)
                # Ensure valid barycentric for this point
                if start_pt[0] + start_pt[1] > 1 - 2e-9:
                    s = (1 - 2e-9) / (start_pt[0] + start_pt[1])
                    start_pt = start_pt * s
                
                try:
                    res = minimize(
                        single_point_obj,
                        start_pt,
                        args=(fixed, i, 100.0),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    new_bary = fixed.copy()
                    new_bary[i] = res.x
                    new_bary = clamp_bary(new_bary)
                    new_val = -hard_min_obj(new_bary.flatten())
                    if new_val > best_pt_val:
                        best_pt_val = new_val
                        best_pt = res.x
                except Exception:
                    pass
            
            bary[i] = best_pt
            bary = clamp_bary(bary)
            new_val = -hard_min_obj(bary.flatten())
            if new_val > current_val:
                current_val = new_val
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """Combined refinement: sequential relocation + graduated soft-min + NM."""
    start_time = time.time()
    bary = bary_init.copy()
    
    # Phase 1: Sequential relocation (structural novelty)
    if time.time() < start_time + time_limit * 0.4:
        bary = sequential_relocation(bary, time_limit=time_limit * 0.4, n_passes=2)
    
    # Phase 2: Graduated soft-min on full 22D
    current_val = -hard_min_obj(bary.flatten())
    bounds = [(0, 1)] * (N * 2)
    
    for beta in [30.0, 100.0, 300.0]:
        if time.time() > start_time + time_limit * 0.7:
            break
        try:
            res = minimize(
                soft_min_obj,
                bary.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            new_bary = clamp_bary(res.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead hard min
    if time.time() < start_time + time_limit * 0.9:
        try:
            res = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
            )
            new_bary = clamp_bary(res.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Phase 4: Final sequential pass
    if time.time() < start_time + time_limit:
        bary = sequential_relocation(bary, time_limit=time_limit * 0.1, n_passes=1)
    
    return bary

def generate_vogel_init():
    """Vogel spiral in barycentric coords - deterministic, well-distributed."""
    # Vogel spiral in unit disk, then map to triangle
    n = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = []
    for i in range(n):
        r = np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Map from unit disk to equilateral triangle via barycentric
        # Simple approach: use (x, y) in [-1,1]^2 mapped to triangle
        # Better: use the fact that the centroid is (1/3, 1/3) in barycentric
        # Map (x,y) from disk to triangle:
        # The triangle in "barycentric" has vertices (1,0,0), (0,1,0), (0,0,1)
        # A point in the disk can be mapped via:
        u = 0.5 + x * 0.4
        v = 0.5 + y * 0.4 - x * 0.2
        w = 1.0 - u - v
        if w < 0:
            # Reflect
            u = 0.5 - x * 0.4
            v = 0.5 + y * 0.4 + x * 0.2
            w = 1.0 - u - v
        if u >= 0 and v >= 0 and w >= 0:
            pts.append([u, v])
        else:
            pts.append([1/3, 1/3])
    
    return np.array(pts)

def generate_hexagonal_init():
    best_bary = None
    best_val = -np.inf
    for scale in [0.18, 0.22, 0.26, 0.30, 0.34, 0.38]:
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
        return generate_vogel_init()
    return best_bary

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
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
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Vogel spiral init
    vogel_init = generate_vogel_init()
    refined_vogel = refine_local(vogel_init, time_limit=12.0)
    update_best(refined_vogel)

    # 2. Hexagonal init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=12.0)
    update_best(refined_hex)

    # 3. Sequential relocation from best (structural core)
    if best_bary is not None and time.time() < deadline - 30:
        seq_refined = sequential_relocation(best_bary, time_limit=15.0, n_passes=4)
        update_best(seq_refined)
        # Polish
        if time.time() < deadline - 15:
            polished = refine_local(best_bary, time_limit=10.0)
            update_best(polished)

    # 4. Multi-start with sequential relocation
    rng = np.random.default_rng(777)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            sigma = 0.04 / (1.0 + 0.15 * restart_count)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=5000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=6000 + restart_count)
        else:
            init = generate_random_init(seed=7000 + restart_count)
        
        # Use sequential relocation as primary refiner
        refined = sequential_relocation(init, time_limit=4.0, n_passes=2)
        update_best(refined)
        
        # If improved, do a full refinement
        if time.time() < deadline - 20:
            refined2 = refine_local(refined, time_limit=5.0)
            update_best(refined2)

    # 5. Final polish
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            res = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
            )
            update_best(clamp_bary(res.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is not None and time.time() < deadline - 3:
        try:
            bounds = [(0, 1)] * (N * 2)
            res = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(res.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is not None and time.time() < deadline - 1:
        # Final sequential pass
        best_bary = sequential_relocation(best_bary, time_limit=1.5, n_passes=1)
        update_best(best_bary)

    cart = bary_to_cart(best_bary)
    return cart
