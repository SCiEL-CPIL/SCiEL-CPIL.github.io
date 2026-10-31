import numpy as np
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)

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

def epigraph_obj(x, t_offset=0.0):
    """
    Epigraph formulation: minimize t such that all triangle areas >= t.
    x = [bary (N*2), t]
    """
    bary = x[:N*2].reshape(N, 2)
    bary = clamp_bary(bary)
    t = x[N*2]
    areas = calc_areas_bary(bary)
    # Penalty for violations: sum of max(0, t - area_i)
    violations = np.maximum(0.0, t - areas)
    return np.sum(violations) + 1e-4 * t  # slight push to increase t

def active_set_ascent(bary_init, time_limit=5.0):
    """
    Coordinated ascent: iteratively move points to increase the smallest triangle areas.
    Uses subgradient-like updates on the active (smallest) triangles.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        # Find active triangles (within tolerance of min)
        tol = max(1e-6, 0.05 * min_area)
        active = np.where(areas <= min_area + tol)[0]
        
        if len(active) == 0:
            break
        
        # Compute gradient of min area w.r.t. barycentric coords
        # For each active triangle, compute partial derivatives
        # Area = |det|, so gradient of det w.r.t. coords
        grad = np.zeros((N, 2))
        
        for tri_i in active:
            i = tri_idx[tri_i, 0]
            j = tri_idx[tri_i, 1]
            k = tri_idx[tri_i, 2]
            
            ui, vi, wi = bary[i, 0], bary[i, 1], 1 - bary[i, 0] - bary[i, 1]
            uj, vj, wj = bary[j, 0], bary[j, 1], 1 - bary[j, 0] - bary[j, 1]
            uk, vk, wk = bary[k, 0], bary[k, 1], 1 - bary[k, 0] - bary[k, 1]
            
            det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
            
            # Partial derivatives of det w.r.t. u_i, v_i, u_j, v_j, u_k, v_k
            # d(det)/du_i = (vj * wk - vk * wj)
            # d(det)/dv_i = (uk * wj - uj * wk)
            # d(det)/du_j = (vi * wk - wi * vk)
            # d(det)/dv_j = (wi * uk - ui * uk)  -> wait, let me recompute
            # Actually: det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
            # d/d ui = vj*wk - vk*wj
            # d/d vi = -(uj*wk - uk*wj) = uk*wj - uj*wk
            # d/d uj = -vi*wk + wi*vk
            # d/d vj = ui*wk - uk*wi  -> wait: d/dvj of [ui*(vj*wk - vk*wj)] = ui*wk
            #         d/dvj of [-vi*(uj*wk - uk*wj)] = 0
            #         d/dvj of [wi*(uj*vk - uk*vj)] = -wi*uk
            #         so d/dvj = ui*wk - wi*uk
            # d/d uk = -vi*wj + wi*vj
            # d/d vk = ui*wj - wi*uj
            
            d_ui = vj * wk - vk * wj
            d_vi = uk * wj - uj * wk
            d_uj = -vi * wk + wi * vk
            d_vj = ui * wk - wi * uk
            d_uk = -vi * wj + wi * vj
            d_vk = ui * wj - wi * uj
            
            sign = np.sign(det) if abs(det) > 1e-15 else 1.0
            
            grad[i, 0] += sign * d_ui
            grad[i, 1] += sign * d_vi
            grad[j, 0] += sign * d_uj
            grad[j, 1] += sign * d_vj
            grad[k, 0] += sign * d_uk
            grad[k, 1] += sign * d_vk
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad /= grad_norm
        
        # Adaptive step size
        base_step = 0.02
        step = base_step / (1 + 0.1 * iteration)
        
        # Try ascent step
        new_bary = clamp_bary(bary + step * grad)
        new_areas = calc_areas_bary(new_bary)
        new_min = np.min(new_areas)
        
        if new_min > min_area + 1e-12:
            bary = new_bary
        else:
            # Try smaller step
            step *= 0.5
            new_bary = clamp_bary(bary + step * grad)
            new_areas = calc_areas_bary(new_bary)
            new_min = np.min(new_areas)
            if new_min > min_area + 1e-12:
                bary = new_bary
            else:
                # Stuck, break
                break
    
    return bary

def epigraph_refine(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with SLSQP.
    """
    start_time = time.time()
    bary = bary_init.copy()
    areas = calc_areas_bary(bary)
    t_init = np.min(areas)
    
    x0 = np.concatenate([bary.flatten(), [t_init]])
    bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
    
    # Constraints: barycentric validity is handled by clamp, but we need
    # u + v <= 1 for each point
    A = np.zeros((N, N * 2 + 1))
    for i in range(N):
        A[i, i * 2] = 1.0
        A[i, i * 2 + 1] = 1.0
    b_lower = np.zeros(N)
    b_upper = np.ones(N)
    constraint = LinearConstraint(A, b_lower, b_upper)
    
    try:
        result = minimize(
            epigraph_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        if result.fun < epigraph_obj(x0):
            new_bary = clamp_bary(result.x[:N * 2].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            old_val = -hard_min_obj(bary.flatten())
            if new_val > old_val:
                return new_bary
    except Exception:
        pass
    
    return bary

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
    
    # Stage 1: Graduated soft-min minimization (beta 20 -> 50 -> 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.35:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Active-set ascent
    bary_current = active_set_ascent(bary_current, time_limit=time_limit * 0.25)
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: Epigraph refinement
    bary_current = epigraph_refine(bary_current, time_limit=time_limit * 0.3)
    
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
    if time.time() < deadline - 35:
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
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 12 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(4.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
        
        # Early exit if we've been stuck for a while
        if restart_count > 15 and best_val > 0.005:
            break

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        # Active-set ascent for fine-tuning
        if time.time() < deadline - 8:
            best_bary = active_set_ascent(best_bary, time_limit=5.0)
            update_best(best_bary)
        
        # Epigraph refinement
        if time.time() < deadline - 4:
            best_bary = epigraph_refine(best_bary, time_limit=3.0)
            update_best(best_bary)
        
        # Final Nelder-Mead polish
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
