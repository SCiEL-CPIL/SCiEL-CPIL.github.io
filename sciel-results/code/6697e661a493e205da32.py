import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
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

def graduated_soft_min_obj(bary_flat, beta_start=10.0, beta_end=300.0, total_iters=100):
    """
    Graduated soft-min: increases beta over iterations to gradually sharpen
    the surrogate toward the true min.
    """
    state = {'iter': 0}
    
    def obj(bary_flat):
        state['iter'] += 1
        progress = min(state['iter'] / total_iters, 1.0)
        # Exponential ramp
        beta = beta_start * (beta_end / beta_start) ** progress
        
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
    
    return obj

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a few scales
    for scale in [0.25, 0.3, 0.35, 0.4]:
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
            for _ in range(3):
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

def generate_lattice_k3_plus1():
    """
    K=3 lattice has 10 points. Add 1 point.
    Lattice points: (i/3, j/3) for i+j <= 3.
    """
    K = 3
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            pts.append([i / K, j / K])
    
    candidates = []
    # Offset 1
    p1 = np.array(pts + [[0.35, 0.35]])
    candidates.append(p1)
    # Offset 2
    p2 = np.array(pts + [[0.45, 0.25]])
    candidates.append(p2)
    # Offset 3
    p3 = np.array(pts + [[0.25, 0.45]])
    candidates.append(p3)
    
    best_bary = None
    best_val = -np.inf
    for c in candidates:
        c_clamped = clamp_bary(c)
        val = -hard_min_obj(c_clamped.flatten())
        if val > best_val:
            best_val = val
            best_bary = c_clamped.copy()
            
    return best_bary

def generate_lattice_k4_minus4():
    """
    K=4 lattice has 15 points. Remove 4.
    """
    K = 4
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            pts.append([i / K, j / K])
            
    pts = np.array(pts)
    
    # Identify interior points (u>0, v>0, u+v<1)
    interior_idx = []
    for i, p in enumerate(pts):
        u, v = p
        if u > 0.01 and v > 0.01 and u + v < 0.99:
            interior_idx.append(i)
            
    # Remove interior points
    remove_idx = set(interior_idx)
    
    # We need to remove 4 points total. We have 3 interior.
    # Remove one corner, e.g., (0,0) which is index 0.
    remove_idx.add(0)
    
    if len(remove_idx) < 4:
        # If we didn't find enough interior, remove another corner
        remove_idx.add(1) # (1,0) is index 1? No, (1,0) is i=1,j=0.
        # Let's just remove index 1 as well if needed.
        
    keep_idx = [i for i in range(len(pts)) if i not in remove_idx]
    kept_pts = pts[keep_idx]
    
    # Ensure we have exactly 11
    if len(kept_pts) != 11:
        # Fallback to hexagonal
        return generate_hexagonal_init()
        
    return clamp_bary(kept_pts)

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
    best_val_local = -hard_min_obj(bary_current.flatten())
    
    def try_update(bary_candidate):
        nonlocal bary_current, best_val_local
        bary_candidate = clamp_bary(bary_candidate)
        val = -hard_min_obj(bary_candidate.flatten())
        if val > best_val_local:
            bary_current = bary_candidate.copy()
            best_val_local = val
            
    # Stage 1: Graduated soft-min with L-BFGS-B
    try:
        if time_limit >= 3.0:
            total_iters = max(50, int(time_limit * 15))
            grad_obj = graduated_soft_min_obj(bary_current.flatten(), 
                                               beta_start=5.0, 
                                               beta_end=300.0,
                                               total_iters=total_iters)
            result1 = minimize(
                grad_obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': total_iters, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            try_update(result1.x.reshape(N, 2))
        else:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            try_update(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.35:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        nm_maxiter = max(500, int(time_limit * 200))
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': nm_maxiter, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
        )
        try_update(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.70:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-16, 'iprint': 0}
        )
        try_update(result3.x.reshape(N, 2))
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

    # 1. Try diverse high-quality initializations
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # K=3 + 1
    k3_init = generate_lattice_k3_plus1()
    inits.append(k3_init)
    
    # K=4 - 4
    k4_init = generate_lattice_k4_minus4()
    inits.append(k4_init)
    
    # Random
    rng_init = np.random.default_rng(0)
    inits.append(generate_random_init(seed=0))
    
    # Refine each initial configuration
    for init in inits:
        if time.time() > deadline - 30:
            break
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)

    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(0, 1)] * (N * 2)
            # Use a larger population and more iterations for better global coverage
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
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
        # Alternate between perturbation and fresh starts
        if best_bary is not None:
            if restart_count % 3 == 0:
                init = generate_perturbed_init(best_bary, sigma=0.08, seed=2000 + restart_count)
            elif restart_count % 3 == 1:
                init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
            else:
                init = generate_random_init(seed=3000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - time.time()
        if remaining > 30:
            tl = 4.0
        elif remaining > 20:
            tl = 3.0
        else:
            tl = 2.0
            
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # SLSQP with very tight tolerance
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-17, 'iprint': 0}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            # Nelder-Mead with extremely tight tolerance
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-18, 'fatol': 1e-22, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
