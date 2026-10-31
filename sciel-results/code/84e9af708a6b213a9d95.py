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

def fairness_obj(bary_flat, gamma=10.0):
    """
    Objective that penalizes variance among the smallest triangles,
    encouraging equalization of the minimum-area triangles.
    Combines soft-min with a penalty on spread of small areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Soft-min component
    beta = 30.0
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min = -np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
    
    # Fairness: penalize variance of the bottom 30% of areas
    sorted_areas = np.sort(areas)
    n_small = max(1, int(0.3 * len(sorted_areas)))
    small_areas = sorted_areas[:n_small]
    spread_penalty = np.var(small_areas) * gamma
    
    return -soft_min + spread_penalty

def equalize_min_obj(bary_flat, t_init=None, gamma=50.0):
    """
    Epigraph-style: maximize t where t is a lower bound on all areas,
    with smooth penalty for violations and bonus for equalizing areas near t.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Use the current min as t estimate, or a fixed fraction
    if t_init is None:
        t = np.min(areas)
    else:
        t = t_init
    
    # Penalty for areas below t
    below = np.maximum(0, t - areas)
    penalty = np.sum(below**2) * gamma
    
    # Bonus: push areas that are close to t upward (equalization)
    near_mask = (areas > t * 0.8) & (areas < t * 1.2)
    if np.any(near_mask):
        near_bonus = np.mean(areas[near_mask]) - t
    else:
        near_bonus = 0.0
    
    return -t + penalty - 0.1 * near_bonus

def epigraph_obj(params):
    """
    Epigraph formulation: minimize t subject to areas >= t.
    params = [bary (N*2), t]
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t
    violation = np.maximum(0, t - areas)
    penalty = np.sum(violation**2) * 1000
    return -t + penalty

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
    Generate a symmetric initial configuration based on known Heilbronn structures.
    """
    # Start with a regular-ish hexagonal pattern adapted to 11 points
    # 3 corners + 3 edge midpoints + 5 interior
    bary = np.array([
        [0.01, 0.01],    # corner 1
        [0.01, 0.98],    # corner 2  
        [0.98, 0.01],    # corner 3
        [0.01, 0.5],     # edge
        [0.5, 0.01],     # edge
        [0.49, 0.49],    # center
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.5]
    ])
    return clamp_bary(bary)

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

def refine_local_fast(bary_init, time_limit=1.0):
    """
    Fast local refinement for use inside SA/ILS loops.
    Uses fewer iterations than the full refine_local.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Quick L-BFGS-B with soft min
    try:
        result = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_bary = clamp_bary(result.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
    
    # Quick Nelder-Mead
    if time.time() - start_time < time_limit:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
            
    return bary_current

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min,
    fairness equalization, and high-precision polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min minimization (multiple passes with increasing beta)
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0]
    for step_idx, beta in enumerate(beta_schedule):
        if time.time() - start_time > time_limit * 0.35:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
    
    # Stage 1.5: Fairness equalization pass
    if time.time() - start_time < time_limit * 0.5:
        try:
            result_fair = minimize(
                fairness_obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result_fair.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.55:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.75:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-15}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
    
    # Stage 4: Final Nelder-Mead micro-polish
    if time.time() - start_time < time_limit * 0.95:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            new_bary = clamp_bary(result4.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    
    # Initial guess for t
    areas = calc_areas_bary(clamp_bary(bary_init))
    t_init = np.min(areas) * 0.5
    
    # Full params: bary (22) + t (1) = 23
    bounds_full = [(0, 1)] * (N * 2) + [(0, 1)]
    
    params_init = np.concatenate([clamp_bary(bary_init).flatten(), [t_init]])
    
    try:
        result = minimize(
            epigraph_obj,
            params_init,
            method='SLSQP',
            bounds=bounds_full,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        new_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
        old_val = -hard_min_obj(bary_init.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            return new_bary
    except Exception:
        pass
    
    return bary_init

def simulated_annealing_ilS(start_bary, time_limit=60.0, seed=42):
    """
    Simulated Annealing with Iterated Local Search.
    Uses a cooling schedule and periodically runs local refinement.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    current_bary = clamp_bary(start_bary)
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    # Initial temperature based on typical step size impact
    T = 0.05 
    cooling_rate = 0.9995
    
    # Adaptive step size
    step_size = 0.1
    
    # Local refinement interval (every X iterations)
    refine_interval = 50
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        # Perturb a random subset of points
        num_perturbed = rng.integers(1, 4) # 1 to 3 points
        indices = rng.choice(N, num_perturbed, replace=False)
        
        new_bary = current_bary.copy()
        for idx in indices:
            # Gaussian perturbation
            new_bary[idx, 0] += rng.normal(0, step_size)
            new_bary[idx, 1] += rng.normal(0, step_size)
        
        new_bary = clamp_bary(new_bary)
        new_val = -hard_min_obj(new_bary.flatten())
        
        # Metropolis acceptance criterion
        delta = new_val - current_val
        if delta > 0:
            current_bary = new_bary
            current_val = new_val
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_bary = new_bary
                current_val = new_val
        
        # Update best
        if current_val > best_val:
            best_val = current_val
            best_bary = current_bary.copy()
            
        # Cooling
        T *= cooling_rate
        
        # Adaptive step size: if we've had many rejections, shrink step; if many accepts, grow slightly
        # Simple heuristic: keep track of last 100 acceptances
        # For now, just decay step size slowly
        step_size *= 0.9998
        if step_size < 0.001:
            step_size = 0.001
            
        # Periodic Local Refinement
        iter_count += 1
        if iter_count % refine_interval == 0:
            # Run a fast local refinement on the current state
            # This helps escape flat regions or sharpen the minimum
            refined = refine_local_fast(current_bary, time_limit=1.0)
            refined_val = -hard_min_obj(refined.flatten())
            
            if refined_val > current_val:
                current_bary = refined
                current_val = refined_val
                if current_val > best_val:
                    best_val = current_val
                    best_bary = current_bary.copy()
            
            # Occasionally restart from best with a larger perturbation
            if rng.random() < 0.1:
                perturbed_best = generate_perturbed_init(best_bary, sigma=0.2, seed=iter_count)
                current_bary = perturbed_best
                current_val = -hard_min_obj(current_bary.flatten())
                T = 0.05 # Reset temperature
                step_size = 0.1 # Reset step size

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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Try symmetric init
    sym_init = generate_symmetric_init()
    refined_sym = refine_local(sym_init, time_limit=10.0)
    update_best(refined_sym)
    
    # 3. Epigraph refinement from best so far
    if best_bary is not None:
        refined_epi = refine_epigraph(best_bary, time_limit=8.0)
        update_best(refined_epi)
    
    # 4. Simulated Annealing / ILS Global Search
    # This replaces the generic DE and random restarts
    # It is designed to explore the landscape more effectively
    if time.time() < deadline - 40:
        try:
            # Run SA/ILS from the best known solution
            sa_bary = simulated_annealing_ilS(best_bary, time_limit=30.0, seed=101)
            update_best(sa_bary)
            
            # Run SA/ILS from a random start to explore a different basin
            if time.time() < deadline - 20:
                random_start = generate_random_init(seed=999)
                sa_bary2 = simulated_annealing_ilS(random_start, time_limit=15.0, seed=102)
                update_best(sa_bary2)
        except Exception:
            pass

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 4:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 1.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Final micro Nelder-Mead
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                result_micro = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-20, 'adaptive': True}
                )
                update_best(clamp_bary(result_micro.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
