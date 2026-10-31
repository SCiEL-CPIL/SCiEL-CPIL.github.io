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

def energy_obj(bary_flat, beta=30.0, gamma=0.5):
    """
    Energy-based objective: combines soft-min with a repulsion term.
    Encourages points to spread out while emphasizing the minimum area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Soft-min component
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min = -np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
    
    # Repulsion: encourage points to be far apart (in barycentric space)
    # Use pairwise distances to avoid clustering
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((bary[i] - bary[j])**2))
            dists[i,j] = d
            dists[j,i] = d
    
    # Sum of inverse distances (penalizes close points)
    np.fill_diagonal(dists, 1.0)
    repulsion = np.sum(1.0 / (dists + 1e-10))
    
    return -soft_min + gamma * repulsion

def get_active_triangles(bary, threshold=1.05):
    """
    Get triangles whose area is close to the minimum (within threshold factor).
    Returns list of (i, j, k, area) tuples.
    """
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active_mask = areas <= min_area * threshold
    active_indices = np.where(active_mask)[0]
    
    active_tris = []
    for idx in active_indices:
        i, j, k = tri_idx[idx]
        active_tris.append((i, j, k, areas[idx]))
    
    return active_tris

def epigraph_obj(x, beta=20.0):
    """
    Epigraph formulation: minimize t subject to all areas >= t.
    x = [bary_coords (N*2), t]
    Uses smooth log-sum-exp constraints for better conditioning.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Smooth penalty: log-sum-exp of violations
    violations = t - areas
    max_viol = np.max(violations)
    
    if max_viol > 700 / beta:
        # Hard fallback
        penalty = np.sum(np.maximum(violations, 0) ** 2)
    else:
        scaled = beta * violations
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        penalty = log_sum / beta
    
    # Small regularization
    reg = 1e-8 * t
    
    return t + penalty + reg

def epigraph_constrained_obj(x, beta=20.0):
    """
    Alternative epigraph: maximize t with log-sum-exp soft constraint.
    x = [bary_coords (N*2), t]
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft constraint: -log(sum(exp(beta*(t - area)))) / beta
    # This is large when any area < t
    scaled = beta * (t - areas)
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Hard constraint violation
        return t + 1e6 * np.sum(np.maximum(t - areas, 0))
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_penalty = log_sum / beta
    
    return t + soft_penalty

def coordinated_ascent_step(bary, step_size=0.02, n_steps=5):
    """
    Perform coordinated ascent: for each active (near-min) triangle,
    try to move its three points to increase the triangle area.
    """
    bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    for step in range(n_steps):
        active_tris = get_active_triangles(bary, threshold=1.1)
        if not active_tris:
            break
        
        for (i, j, k, area) in active_tris[:10]:  # Limit to top 10 most critical
            # Compute gradient-like direction to increase area of triangle (i,j,k)
            # Area = 0.5 * |cross(pj - pi, pk - pi)|
            # To increase area, move points in directions that increase cross product
            
            pi = bary[i].copy()
            pj = bary[j].copy()
            pk = bary[k].copy()
            
            # Current area vector (2D cross product magnitude)
            v1 = pj - pi
            v2 = pk - pi
            cross_val = v1[0] * v2[1] - v1[1] * v2[0]
            area_sign = np.sign(cross_val) if cross_val != 0 else 1
            
            # Perturb each point in a direction that increases the area
            for pt_idx in [i, j, k]:
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        trial = bary.copy()
                        trial[pt_idx] += [dx, dy]
                        trial = clamp_bary(trial)
                        
                        trial_val = -hard_min_obj(trial.flatten())
                        if trial_val > best_val:
                            bary = trial.copy()
                            best_val = trial_val
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with epigraph formulation.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Epigraph formulation with SLSQP
    try:
        # Initialize t to current min area
        init_areas = calc_areas_bary(bary_current)
        init_t = np.min(init_areas) * 0.9
        
        x_init = np.concatenate([bary_current.flatten(), [init_t]])
        bounds_full = bounds_bary + [(0, 1)]  # t is bounded
        
        result1 = minimize(
            epigraph_obj,
            x_init,
            method='SLSQP',
            bounds=bounds_full,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if result1.success:
            bary_candidate = clamp_bary(result1.x[:N*2].reshape(N, 2))
            val_candidate = -hard_min_obj(bary_candidate.flatten())
            val_current = -hard_min_obj(bary_current.flatten())
            if val_candidate > val_current:
                bary_current = bary_candidate
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Coordinated ascent on active triangles
    try:
        bary_current = coordinated_ascent_step(bary_current, step_size=0.015, n_steps=3)
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 3: Soft-min refinement with L-BFGS-B
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(80.0,),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        bary_candidate = clamp_bary(result2.x.reshape(N, 2))
        val_candidate = -hard_min_obj(bary_candidate.flatten())
        val_current = -hard_min_obj(bary_current.flatten())
        if val_candidate > val_current:
            bary_current = bary_candidate
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_candidate = clamp_bary(result3.x.reshape(N, 2))
        val_candidate = -hard_min_obj(bary_candidate.flatten())
        val_current = -hard_min_obj(bary_current.flatten())
        if val_candidate > val_current:
            bary_current = bary_candidate
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current
    
    # Stage 5: SLSQP polish
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds_bary,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_candidate = clamp_bary(result4.x.reshape(N, 2))
        val_candidate = -hard_min_obj(bary_candidate.flatten())
        val_current = -hard_min_obj(bary_current.flatten())
        if val_candidate > val_current:
            bary_current = bary_candidate
    except Exception:
        pass
        
    return bary_current

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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
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
                maxiter=50,
                popsize=18,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.03 + 0.02 * (restart_count % 3)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Moderate perturbation
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=2500 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.5)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        try:
            # Epigraph refinement with tighter tolerance
            init_areas = calc_areas_bary(best_bary)
            init_t = np.min(init_areas) * 0.98
            
            x_init = np.concatenate([best_bary.flatten(), [init_t]])
            bounds_full = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_epi = minimize(
                epigraph_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_full,
                options={'maxiter': 800, 'ftol': 1e-14}
            )
            
            bary_candidate = clamp_bary(result_epi.x[:N*2].reshape(N, 2))
            update_best(bary_candidate)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 7:
            # High-beta soft-min polish
            try:
                result_soft = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                update_best(clamp_bary(result_soft.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 5:
            # Coordinated ascent on best solution
            try:
                bary_candidate = coordinated_ascent_step(best_bary, step_size=0.005, n_steps=10)
                update_best(bary_candidate)
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 2000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 1.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Final SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.3:
            try:
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 3000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
