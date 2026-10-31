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
    
    # Try a range of scales to find a good starting point
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

def generate_voronoi_init(seed=None):
    """
    Generate a well-spaced initial configuration using a greedy farthest-point
    sampling approach within the triangle (Voronoi-like).
    """
    rng = np.random.default_rng(seed)
    
    # Start with a point near the centroid
    bary = np.zeros((N, 2))
    bary[0] = [1/3, 1/3]
    
    for i in range(1, N):
        # Generate many candidates and pick the one that maximizes the minimum
        # distance to already-placed points
        best_candidate = None
        best_min_dist = -1
        
        for _ in range(200):
            u = rng.random()
            v = rng.random() * (1 - u)
            candidate = np.array([u, v])
            
            # Compute min distance to existing points (in barycentric space)
            dists = np.sqrt(np.sum((bary[:i] - candidate) ** 2, axis=1))
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate
        
        bary[i] = best_candidate
    
    return clamp_bary(bary)

def high_precision_epigraph_polish(bary_init, time_limit=10.0):
    """
    High-precision epigraph polish using very high beta soft-min approximation.
    This provides a smoother surrogate that closely approximates the hard min,
    potentially escaping shallow local optima.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    bounds_bary = [(0, 1)] * (N * 2)
    
    # Use very high beta values for near-exact min approximation
    betas = [500.0, 1000.0]
    for beta in betas:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success:
                new_bary = clamp_bary(result.x.reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                old_val = -hard_min_obj(bary_current.flatten())
                if new_val > old_val:
                    bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.75:
        return bary_current
    
    # Final Nelder-Mead with very tight tolerances
    try:
        result_nm = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-17, 'adaptive': True}
        )
        new_bary = clamp_bary(result_nm.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        old_val = -hard_min_obj(bary_current.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
    
    return bary_current

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    Uses graduated soft-min and epigraph formulation.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min with increasing beta
    betas = [20.0, 50.0, 100.0, 200.0]
    for beta in betas:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.success or result.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
        
    # Stage 2: Epigraph formulation
    if time.time() - start_time < time_limit * 0.8:
        try:
            # Initialize t with current min area
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.9
            params_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_epigraph = bounds_bary + [(0.0, 1.0)]
            
            def epigraph_obj(params):
                bary_flat = params[:N*2]
                t = params[N*2]
                
                bary = bary_flat.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                
                # Penalty for violating areas >= t
                violations = np.maximum(0, t - areas)
                penalty = 100.0 * np.sum(violations**2)
                
                # Objective: minimize -t + penalty
                return -t + penalty
            
            result_ep = minimize(
                epigraph_obj,
                params_init,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result_ep.success:
                new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                old_val = -hard_min_obj(bary_current.flatten())
                if new_val > old_val:
                    bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.85:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
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
        
    if time.time() - start_time > time_limit * 0.95:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds_bary,
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
    refined_hex = refine_local(hex_init, time_limit=7.0)
    update_best(refined_hex)
    
    # 1b. Try Voronoi-like Init for better spacing
    if time.time() < deadline - 60:
        vor_init = generate_voronoi_init(seed=42)
        refined_vor = refine_local(vor_init, time_limit=7.0)
        update_best(refined_vor)
    
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
                maxiter=35,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=7.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 18:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        elif restart_count % 3 == 0:
            # Voronoi-like restart for diversity
            init = generate_voronoi_init(seed=5000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 18 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(4.5, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 3:
        # High-precision epigraph polish
        try:
            polished = high_precision_epigraph_polish(best_bary, time_limit=min(12.0, deadline - time.time() - 2))
            update_best(polished)
        except Exception:
            pass
        
        # Final SLSQP + Nelder-Mead polish
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-17, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
