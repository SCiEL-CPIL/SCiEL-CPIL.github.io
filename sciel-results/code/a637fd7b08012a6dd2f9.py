"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

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

def clamp_bary(bary, eps=1e-6):
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

def soft_min_obj(bary_flat, beta=80.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def power_sum_obj(bary_flat, p=8.0):
    """
    Smooth surrogate using sum of inverse powers.
    Heavily penalizes small areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Add small epsilon to avoid division by zero
    areas_safe = areas + 1e-12
    return np.sum(areas_safe ** (-p))

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def constrained_obj(x):
    """
    Objective for SLSQP: minimize -t
    x = [bary_flat (22 vars), t (1 var)]
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -t

def constrained_constraints(x):
    """
    Constraints for SLSQP: areas_i >= t - tol
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # We want areas >= t - small_tol
    # SLSQP constraints are in form c(x) >= 0
    # So areas - (t - tol) >= 0  =>  areas - t + tol >= 0
    tol = 1e-6
    return areas - t + tol

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
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

def generate_perturbed_init(base_bary, sigma=0.08, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using a 3-stage process:
    1. L-BFGS-B on Power-Sum or Log-Sum-Exp surrogate.
    2. SLSQP with epigraph constraints (ratcheting the min area).
    3. Nelder-Mead on hard min for final polish.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    bounds_t = [(0, 1)] # t is area, max area is 1/sqrt(3) ~ 0.577, so 1 is safe upper bound
    
    # Stage 1: Smooth Descent
    # Try Power Sum first as it often creates steeper gradients for the min
    try:
        result1 = minimize(
            power_sum_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
    except Exception:
        result1 = minimize(
            soft_min_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds_bary,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
    
    bary_mid = clamp_bary(result1.x.reshape(N, 2))
    
    if time.time() - start_time > time_limit * 0.4:
        # If stage 1 took too long, skip to a quick polish
        try:
            result3 = minimize(
                hard_min_obj,
                bary_mid.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            return clamp_bary(result3.x.reshape(N, 2))
        except Exception:
            return bary_mid

    # Stage 2: Constrained SLSQP (Ratcheting)
    # Initialize t with current min area
    areas_mid = calc_areas_bary(bary_mid)
    t_init = np.min(areas_mid)
    x_init = np.concatenate([bary_mid.flatten(), [t_init]])
    bounds_full = bounds_bary + bounds_t
    
    try:
        # We want to maximize t, so minimize -t
        # Constraints: areas >= t - tol
        result2 = minimize(
            constrained_obj,
            x_init,
            method='SLSQP',
            bounds=bounds_full,
            constraints=[{'type': 'ineq', 'fun': constrained_constraints}],
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if result2.success or result2.fun < -t_init: # Check if t increased
             bary_slsqp = clamp_bary(result2.x[:N*2].reshape(N, 2))
        else:
             bary_slsqp = bary_mid
             
    except Exception:
        bary_slsqp = bary_mid
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_slsqp

    # Stage 3: Nelder-Mead Polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_slsqp.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        return clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        return bary_slsqp

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
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
    
    # 2. Global Search with Differential Evolution (shorter budget)
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            # Reduced population and iterations to save time for local refinement
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20,
                popsize=8,
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
    # Use remaining time for multi-start
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.05, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
        
        # If we are running low on time, break
        if time.time() > deadline - 20:
            break

    # 4. Final high-precision polish: SLSQP then Nelder-Mead
    if best_bary is not None and time.time() < deadline - 3:
        try:
            # Final SLSQP ratchet
            areas_best = calc_areas_bary(best_bary)
            t_init = np.min(areas_best)
            x_init = np.concatenate([best_bary.flatten(), [t_init]])
            bounds_full = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_final_slsqp = minimize(
                constrained_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_full,
                constraints=[{'type': 'ineq', 'fun': constrained_constraints}],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result_final_slsqp.fun < -t_init:
                 update_best(clamp_bary(result_final_slsqp.x[:N*2].reshape(N, 2)))
                 
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Safety check to ensure points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
    
    return cart
