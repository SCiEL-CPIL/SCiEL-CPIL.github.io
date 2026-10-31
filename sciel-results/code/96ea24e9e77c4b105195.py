"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x,y) to barycentric (u,v)."""
    w = cart[:, 1] / H
    v = cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def hybrid_obj(bary_flat, beta=100.0, lam=0.02, gamma=50.0):
    """
    Hybrid objective: soft-min of areas + penalty on variance of smallest areas.
    This regularizes the landscape to encourage balanced critical triangles.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft min of all areas
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min = np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
    
    # Variance penalty on the smallest areas
    k = min(20, len(areas))
    small = np.sort(areas)[:k]
    sm = small.min()
    sh2 = np.clip(-gamma * (small - sm), -500, 500)
    lse2 = np.log(np.sum(np.exp(sh2)))
    var_soft = sm - lse2 / gamma
    
    return -(soft_min - lam * var_soft)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration with random offsets.
    """
    best_bary = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    for s in np.linspace(0.1, 0.5, 15):
        for _ in range(3):
            offset_u = rng.random() * 0.1
            offset_v = rng.random() * 0.1
            points = []
            for i in range(6):
                for j in range(6):
                    u = i * s + offset_u
                    v = j * s * np.sqrt(3) / 2 + offset_v
                    if 0 <= u and 0 <= v and u + v <= 1 and len(points) < N:
                        points.append([u, v])
            
            if len(points) == N:
                bary = np.array(points)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def langevin_dynamics(bary_init, time_budget, rng):
    """
    Physics-based Langevin dynamics with simulated annealing.
    Each point is a particle with potential defined by inverse of min triangle area.
    Thermal fluctuations help escape local minima.
    """
    start_time = time.time()
    cur_bary = bary_init.copy()
    cur_score = -hard_min_obj(cur_bary.flatten())
    best_bary = cur_bary.copy()
    best_score = cur_score
    
    T = 0.015  # Initial temperature
    step = 0.04  # Initial step size
    
    while time.time() - start_time < time_budget:
        # Generate random perturbation
        move = rng.normal(0, step, size=(N, 2))
        cand = clamp_bary(cur_bary + move)
        cand_score = -hard_min_obj(cand.flatten())
        
        # Metropolis acceptance criterion
        delta = cand_score - cur_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            cur_bary = cand
            cur_score = cand_score
            if cur_score > best_score:
                best_score = cur_score
                best_bary = cur_bary.copy()
        
        # Cool down
        T *= 0.97
        step *= 0.99
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_bary = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize(bary_init, method='L-BFGS-B', beta1=80.0, beta2=200.0, 
                     maxiter1=200, maxiter2=300, use_hybrid=True):
        """
        Two-stage optimization + Nelder-Mead polish.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration with hybrid or soft-min objective
            if use_hybrid:
                def obj1(x):
                    return hybrid_obj(x, beta=beta1, lam=0.02, gamma=50.0)
            else:
                def obj1(x):
                    return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            current_best = result1.x.reshape(N, 2)
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(current_best)
                return

            # Stage 2: Exploitation
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                current_best.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            current_best = result2.x.reshape(N, 2)
            
            # Nelder-Mead Polish
            if time.time() < deadline - 2:
                bary_nm = current_best.copy()
                bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
                bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
                
                res_nm = minimize(
                    hard_min_obj,
                    bary_nm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
                )
                current_best = res_nm.x.reshape(N, 2)
            
            update_best(current_best)
                
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=80.0, beta2=200.0, 
                 maxiter1=300, maxiter2=400, use_hybrid=True)
    
    # 2. Langevin dynamics from the best found so far
    if best_bary is not None and time.time() < deadline - 30:
        ld_bary = langevin_dynamics(best_bary, time_budget=20.0, rng=rng)
        update_best(ld_bary)
    
    # 3. Try multiple random initializations
    num_random_starts = 6
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, method='L-BFGS-B', beta1=80.0, beta2=150.0, 
                     maxiter1=150, maxiter2=200, use_hybrid=(i % 2 == 0))
    
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(4):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=300.0, 
                         maxiter1=200, maxiter2=300, use_hybrid=True)
    
    # 5. Another Langevin pass from the best
    if best_bary is not None and time.time() < deadline - 15:
        ld_bary = langevin_dynamics(best_bary, time_budget=12.0, rng=rng)
        update_best(ld_bary)
    
    # 6. Final polish: high-precision L-BFGS-B with very high beta
    if best_bary is not None and time.time() < deadline - 10:
        try:
            def obj_final(x):
                return soft_min_obj(x, beta=500.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_final = minimize(
                obj_final,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            update_best(result_final.x.reshape(N, 2))
        except Exception:
            pass
    
    # 7. Final Nelder-Mead polish on hard min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_nm = best_bary.copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            final_val = -hard_min_obj(final_bary.flatten())
            
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary.copy()
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
            
    return cart
