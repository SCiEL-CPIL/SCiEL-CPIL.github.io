"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)

# Triangle vertices in Cartesian
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H), w = 1-u-v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - y / (2 * H)
    u = 1 - x - y / (2 * H)
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of shape (165,) containing the area ratios.
    """
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

def hard_min_obj(bary_flat):
    """Objective: negative of minimum triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """Smooth approximation of the min using log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid numerical issues with very small areas
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    # Overflow protection
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    return soft_min_val

def generate_structured_init():
    """
    Generate a structured initial configuration based on known good patterns.
    For 11 points, use rows within the triangle.
    """
    points = []
    # Bottom row: 4 points
    for i in range(4):
        x = (i + 0.5) / 4.0
        points.append([x, 0.0])
    
    # Middle row: 3 points
    mid_y = H / 2.0
    for i in range(3):
        x = 0.25 + i * 0.25
        points.append([x, mid_y])
    
    # Upper row: 2 points
    up_y = 2 * H / 3.0
    for i in range(2):
        x = 0.25 + i * 0.5
        points.append([x, up_y])
    
    # Top: 1 point
    points.append([0.5, H])
    
    cart = np.array(points)
    bary = cart_to_bary(cart)
    return clamp_bary(bary)

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

def refine_local(bary_init, time_limit=10.0):
    """
    Refine a configuration using multiple optimization stages.
    1. L-BFGS-B on soft min (smooth landscape)
    2. Nelder-Mead on hard min (robust local search)
    3. L-BFGS-B on hard min (precise polishing)
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    if time.time() - start_time < time_limit * 0.3:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.success or result1.fun < soft_min_obj(bary_current.flatten(), 50.0):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 2: Hard min refinement with Nelder-Mead
    if time.time() - start_time < time_limit * 0.7:
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
            
    # Stage 3: Hard min polishing with L-BFGS-B
    if time.time() - start_time < time_limit * 0.95:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            bary_current = clamp_bary(result3.x.reshape(N, 2))
        except Exception:
            pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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

    # Phase 1: Global Search with Differential Evolution
    # This is the key structural change to escape local minima.
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            
            # Tuned for speed and exploration
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                mutation=(0.4, 1.0),
                recombination=0.9
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            # Refine the DE result
            refined_de = refine_local(de_bary, time_limit=15.0)
            update_best(refined_de)
        except Exception:
            pass

    # Phase 2: Multi-start Local Refinement
    # Try structured init and random starts
    
    # 2a: Structured Init
    if time.time() < deadline - 20:
        structured_init = generate_structured_init()
        refined_struct = refine_local(structured_init, time_limit=10.0)
        update_best(refined_struct)

    # 2b: Random Starts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        # Alternate between random and perturbed best
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.05, seed=2000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
            
        # Shorter refinement for more restarts
        refined = refine_local(init, time_limit=5.0)
        update_best(refined)

    # Phase 3: Final High-Precision Polish on Best Solution
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # Use SLSQP for final constraint handling and precision if available and fast
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            final_bary = clamp_bary(result_final.x.reshape(N, 2))
            update_best(final_bary)
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
