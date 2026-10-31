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
V2 = np.array([0.5, np.sqrt(3) / 2)
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
    Uses soft constraints for smoothness.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft penalty for violations: sum of (t - area)^2 for areas < t
    violations = np.maximum(t - areas, 0)
    penalty = np.sum(violations ** 2)
    
    # Also add a small regularization to keep t reasonable
    reg = 1e-6 * t
    
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
    Combines the 5-stage graduated soft-min approach from the inspiration with
    epigraph and coordinated ascent from the current program.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
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
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
    
    # Stage 3: Sharp soft-min (high beta)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.55:
        return bary_current
    
    # Stage 4: Epigraph refinement with SLSQP
    try:
        init_areas = calc_areas_bary(bary_current)
        init_t = np.min(init_areas) * 0.9
        
        x_init = np.concatenate([bary_current.flatten(), [init_t]])
        bounds_full = bounds_bary + [(0, 1)]
        
        result4 = minimize(
            epigraph_obj,
            x_init,
            method='SLSQP',
            bounds=bounds_full,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if result4.success:
            bary_candidate = clamp_bary(result4.x[:N*2].reshape(N, 2))
            val_candidate = -hard_min_obj(bary_candidate.flatten())
            val_current = -hard_min_obj(bary_current.flatten())
            if val_candidate > val_current:
                bary_current = bary_candidate
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 5: Coordinated ascent on active triangles
    try:
        bary_current = coordinated_ascent_step(bary_current, step_size=0.015, n_steps=3)
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.85:
        return bary_current
    
    # Stage 6: Hard min refinement with Nelder-Mead
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_candidate = clamp_bary(result5.x.reshape(N, 2))
        val_candidate = -hard_min_obj(bary_candidate.flatten())
        val_current = -hard_min_obj(bary_current.flatten())
        if val_candidate > val_current:
            bary_current = bary_candidate
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.95:
        return bary_current
    
    # Stage 7: SLSQP polish
    try:
        result6 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds_bary,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_candidate = clamp_bary(result6.x.reshape(N, 2))
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

def simulated_annealing(bary_init, time_limit=100.0, seed=42):
    """
    Structural change: Simulated Annealing with constraint-aware moves.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    current_val = best_val
    
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Pre-compute indices for faster access
    # We will pick a random triangle index, then pick a vertex to move
    
    while time.time() < deadline:
        # Generate a move
        # Strategy: 70% chance pick a "bad" triangle (small area) and move one of its vertices
        # 30% chance random move
        
        if rng.random() < 0.7:
            # Pick a triangle proportional to "badness" (small area)
            areas = calc_areas_bary(bary)
            # Inverse area as weight? Or just pick from smallest K?
            # Let's pick from the smallest 20 triangles
            min_area = np.min(areas)
            # Threshold to define "active"
            active_mask = areas <= min_area * 1.5
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 0:
                tri_idx_local = rng.choice(active_indices)
                i, j, k = tri_idx[tri_idx_local]
                # Pick one of the 3 vertices
                pt_idx = rng.choice([i, j, k])
            else:
                pt_idx = rng.integers(0, N)
                tri_idx_local = rng.integers(0, len(tri_idx))
                i, j, k = tri_idx[tri_idx_local]
        else:
            pt_idx = rng.integers(0, N)
            i, j, k = tri_idx[rng.integers(0, len(tri_idx))]
        
        # Generate a small step
        # Move in barycentric coordinates
        # To ensure we stay inside, we can move towards the centroid of the triangle formed by the other 2 points and the vertex?
        # Or just a random Gaussian step.
        step_size = 0.02 * (1 + T) # Larger steps at higher T
        delta_u = rng.normal(0, step_size)
        delta_v = rng.normal(0, step_size)
        
        new_bary = bary.copy()
        new_bary[pt_idx, 0] += delta_u
        new_bary[pt_idx, 1] += delta_v
        new_bary = clamp_bary(new_bary)
        
        new_val = -hard_min_obj(new_bary.flatten())
        
        # Acceptance criterion
        if new_val > current_val:
            bary = new_bary
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_bary = bary.copy()
        else:
            # Accept with probability exp((new_val - current_val) / T)
            # Note: new_val and current_val are negative (minimization of -area)
            # So new_val - current_val is negative if we got worse.
            delta = new_val - current_val
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    bary = new_bary
                    current_val = new_val
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
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

    # 1. Try Hexagonal Init and refine
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=5.0)
    update_best(refined_hex)
    
    # 2. Run Simulated Annealing for the majority of the time
    # This is the structural change: replacing the multi-start random/DE loop with SA
    if time.time() < deadline - 10:
        sa_time = min(100.0, deadline - time.time() - 10)
        sa_bary = simulated_annealing(hex_init, time_limit=sa_time, seed=42)
        update_best(sa_bary)
        
        # 3. Refine the SA result
        if time.time() < deadline - 5:
            refined_sa = refine_local(sa_bary, time_limit=5.0)
            update_best(refined_sa)
            
        # 4. If time remains, do a few more SA runs from different inits
        if time.time() < deadline - 10:
            for seed in [123, 456]:
                if time.time() >= deadline - 10: break
                random_init = generate_random_init(seed=seed)
                sa_bary2 = simulated_annealing(random_init, time_limit=min(20.0, deadline - time.time() - 10), seed=seed)
                update_best(sa_bary2)
                if time.time() < deadline - 5:
                    refined_sa2 = refine_local(sa_bary2, time_limit=5.0)
                    update_best(refined_sa2)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            # Quick local refinement
            final_refined = refine_local(best_bary, time_limit=min(5.0, deadline - time.time() - 1))
            update_best(final_refined)
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
