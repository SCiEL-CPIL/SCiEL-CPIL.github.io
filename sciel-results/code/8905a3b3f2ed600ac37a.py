import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    S = 1 - y / H
    u = 0.5 + 0.5 * S - x
    v = S - u
    return np.column_stack([u, v])

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

def calc_min_area(bary):
    """Calculate minimum area efficiently."""
    return np.min(calc_areas_bary(bary))

def clamp_bary(bary, eps=1e-10):
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

def generate_hex_init():
    """
    Generate a high-quality hexagonal lattice initialization.
    This is a strong prior for maximizing minimum distances/areas.
    """
    inits = []
    
    for scale in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        center = np.array([1.0/3, 1.0/3])
        cart_center = bary_to_cart(center.reshape(1,2))[0]
        
        cart_pts = [cart_center.copy()]
        
        # 6 points at distance scale
        for k in range(6):
            theta = np.pi/3 * k
            dx = scale * np.cos(theta)
            dy = scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        # Add points from second ring to reach 11
        for k in range(6):
            if len(cart_pts) >= N:
                break
            theta = np.pi/3 * k + np.pi/6
            dx = 2 * scale * np.cos(theta)
            dy = 2 * scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        if len(cart_pts) >= N:
            cart_pts = np.array(cart_pts[:N])
            bary_pts = cart_to_bary(cart_pts)
            bary_pts = clamp_bary(bary_pts)
            
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)
                
    return inits

def generate_fibonacci_init():
    """Generate points using Fibonacci spiral in the triangle."""
    inits = []
    rng = np.random.default_rng(42)
    
    for trial in range(5):
        # Use golden ratio spiral
        phi = (1 + np.sqrt(5)) / 2
        n = N
        r = np.sqrt(np.arange(n) / n)
        theta = 2 * np.pi * np.arange(n) / phi**2 + rng.uniform(0, 2*np.pi)
        
        # Map to triangle using barycentric coordinates
        # Use a simple mapping: center at (1/3, 1/3), scale to fit
        center = np.array([1.0/3, 1.0/3])
        max_r = 0.35
        
        x = center[0] + r * max_r * np.cos(theta)
        y = center[1] + r * max_r * np.sin(theta)
        
        bary = cart_to_bary(np.column_stack([x, y]))
        bary = clamp_bary(bary)
        
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
            
    return inits

def generate_grid_init():
    """Generate points on a triangular grid."""
    inits = []
    
    # Try different grid configurations
    for rows in [3, 4]:
        for cols in [3, 4]:
            pts = []
            for i in range(rows):
                for j in range(cols):
                    if len(pts) >= N:
                        break
                    # Position in barycentric
                    u = (j + 0.5) / cols
                    v = (i + 0.5) / rows * (1 - u)
                    pts.append([u, v])
                    
            if len(pts) >= N:
                bary = np.array(pts[:N])
                bary = clamp_bary(bary)
                inits.append(bary)
                
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.7:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-20, 'adaptive': True}
            )
            new_bary = clamp_bary(result3.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 4: Local search with random perturbations
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            perturbation = rng.normal(0, 0.01, (N, 2))
            new_bary = clamp_bary(bary_current + perturbation)
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
                
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. High-quality Hexagonal starts
    hex_inits = generate_hex_init()
    inits.extend(hex_inits)
    
    # 2. Fibonacci spiral starts
    fib_inits = generate_fibonacci_init()
    inits.extend(fib_inits)
    
    # 3. Grid starts
    grid_inits = generate_grid_init()
    inits.extend(grid_inits)
    
    # 4. Random starts
    for i in range(15):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 5. Perturbed Hex starts
    for hex_init in hex_inits[:4]:
        for _ in range(8):
            perturbed = hex_init + rng.normal(0, 0.03, (N, 2))
            perturbed = clamp_bary(perturbed)
            inits.append(perturbed)

    # Deduplicate inits (approximately)
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.015):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Sort inits by initial quality (best first)
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each initial point with time budget
    for i, init in enumerate(inits):
        if time.time() > deadline - 12:
            break
            
        # Allocate time based on rank and remaining time
        remaining_inits = len(inits) - i
        # Give more time to top-ranked inits
        if i < 3:
            base_time = 6.0
        elif i < 8:
            base_time = 4.0
        else:
            base_time = 2.5
            
        time_per_init = min(base_time, (deadline - time.time() - 12) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 25:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=100,
                tol=1e-12,
                popsize=25,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    # Refine the DE result
                    if time.time() < deadline - 8:
                        refined_de = refine_local(bary_de, time_limit=min(8.0, deadline - time.time() - 4))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 6:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-16, 'fatol': 1e-21, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
