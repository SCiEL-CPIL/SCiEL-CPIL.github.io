import numpy as np
from scipy.optimize import minimize, differential_evolution, powell
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

def objective_func(bary_flat, beta=100.0):
    """
    Objective function using a soft-min (log-sum-exp) approach.
    This is smoother than hard min and helps gradient-based methods.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # If any area is effectively zero, return a large penalty
    if np.min(areas) < 1e-12:
        return 1e10
        
    # Soft min: -1/beta * log(sum(exp(-beta * area)))
    # This approximates min(area) as beta -> infinity
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # Numerical stability
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    return -soft_min_val

def generate_inits():
    """
    Generate high-quality initial configurations.
    1. Symmetric 3-fold patterns.
    2. Fibonacci-like spiral in triangle.
    3. Random with symmetry bias.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Symmetric: Center + 3 rings of 3
    # The optimal solution for 11 points in a triangle is known to have 3-fold symmetry.
    # Structure: 1 center, 3 points at r1, 3 points at r2, 3 points at r3, 1 point?
    # Actually, 11 = 1 + 3 + 3 + 3 + 1? No, that breaks symmetry.
    # Maybe 1 center, 3 points near midpoints, 3 points near vertices, 3 points in between, 1 point?
    # Let's try 1 center, 3 points at 1/3 height, 3 points at 2/3 height, 3 points near top?
    
    # Better: Use the known "best packing" intuition.
    # Points often lie on a triangular lattice.
    
    # Pattern A: Triangular lattice subset
    # Generate a fine grid and pick 11 points? No, let's use parametric symmetry.
    
    # We define points by (radius, angle) relative to centroid (1/3, 1/3) in barycentric?
    # It's easier to work in Cartesian centered at centroid.
    # Centroid C = (0.5, sqrt(3)/6)
    
    cx = 0.5
    cy = np.sqrt(3) / 6
    
    def make_symmetric_init(r1, r2, r3, r4, angle_offset=0.0, center_jitter=0.0):
        """
        Creates 11 points:
        1 at center (with jitter)
        3 at radius r1, angles 0, 120, 240 + offset
        3 at radius r2, angles 60, 180, 300 + offset
        3 at radius r3, angles 0, 120, 240 + offset
        1 at radius r4, angle 0 + offset (breaks full symmetry slightly or acts as 4th ring)
        """
        bary = np.zeros((N, 2))
        
        # Center
        bary[0] = cart_to_bary(np.array([[cx + center_jitter, cy + center_jitter * 0.5]]))[0]
        
        # Ring 1
        for i in range(3):
            theta = 2 * np.pi * i / 3 + angle_offset
            x = cx + r1 * np.cos(theta)
            y = cy + r1 * np.sin(theta)
            bary[1+i] = cart_to_bary(np.array([[x, y]]))[0]
            
        # Ring 2
        for i in range(3):
            theta = 2 * np.pi * i / 3 + np.pi/3 + angle_offset
            x = cx + r2 * np.cos(theta)
            y = cy + r2 * np.sin(theta)
            bary[4+i] = cart_to_bary(np.array([[x, y]]))[0]
            
        # Ring 3
        for i in range(3):
            theta = 2 * np.pi * i / 3 + angle_offset
            x = cx + r3 * np.cos(theta)
            y = cy + r3 * np.sin(theta)
            bary[7+i] = cart_to_bary(np.array([[x, y]]))[0]
            
        # Point 11: Let's put it at a specific spot, maybe slightly off-center or on a ring?
        # To maintain 3-fold symmetry, we can't have a single unique point unless it's at center.
        # So let's make Ring 4 have 3 points? 1+3+3+3+3 = 13. Too many.
        # 1 + 3 + 3 + 4? No.
        # Maybe the 11th point is at the centroid, and the other 10 are 2 rings of 5? No, 5 doesn't fit 3-fold.
        # Maybe 1 center, 3 points, 3 points, 3 points, and 1 point that is actually the centroid?
        # If we have 1 center, we have 10 left. 10 is not divisible by 3.
        # So strict 3-fold symmetry is impossible for 11 points unless one point is at the center and the others form orbits of 3?
        # 1 (center) + 3*3 (9 points) = 10. We have 11.
        # So we must break symmetry or have a different structure.
        # Common structure: 3 vertices of a smaller triangle, 3 midpoints, 3 more, 1 center, 1 more?
        
        # Let's just put the 11th point at a random location near the center to break symmetry slightly,
        # or assume the optimizer will move it.
        bary[10] = cart_to_bary(np.array([[cx + r4 * np.cos(angle_offset), cy + r4 * np.sin(angle_offset)]]))[0]
        
        return clamp_bary(bary)
    
    # Generate variations of symmetric inits
    for r1 in [0.1, 0.15, 0.2]:
        for r2 in [0.2, 0.25, 0.3]:
            for r3 in [0.3, 0.35, 0.4]:
                for r4 in [0.05, 0.1]:
                    for ang in [0.0, 0.1, 0.2]:
                        init = make_symmetric_init(r1, r2, r3, r4, ang)
                        inits.append(init)
                        
    # Add random inits
    for _ in range(20):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using a multi-stage strategy.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Stage 1: L-BFGS-B with soft min (increasing beta)
    for beta in [20.0, 100.0, 500.0, 2000.0]:
        if time.time() - start_time > time_limit * 0.3:
            break
        try:
            result1 = minimize(
                objective_func,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 2: Powell's Method (derivative-free, good for non-smooth min)
    if time.time() - start_time < time_limit * 0.6:
        try:
            def powell_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -calc_min_area(bary)
                
            result2 = powell(
                powell_obj,
                bary_current.flatten(),
                bounds=bounds,
                maxiter=500,
                options={'maxfev': 5000}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for fine tuning
    if time.time() - start_time < time_limit * 0.8:
        try:
            def nm_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -calc_min_area(bary)
                
            result3 = minimize(
                nm_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = clamp_bary(result3.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    # Stage 4: Local Perturbation Search
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.02
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for _ in range(10):
                perturbation = rng.normal(0, step_size, (N, 2))
                new_bary = clamp_bary(bary_current + perturbation)
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step_size *= 0.8
            if step_size < 1e-6:
                break
                
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

    # Generate inits
    inits = generate_inits()
    
    # Sort by initial quality
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine top inits
    num_to_refine = min(len(inits), 20) # Limit to 20 best inits
    
    for i in range(num_to_refine):
        if time.time() > deadline - 15:
            break
            
        # Allocate time
        remaining_inits = num_to_refine - i
        if i < 5:
            time_per_init = 4.0
        elif i < 10:
            time_per_init = 3.0
        else:
            time_per_init = 2.0
            
        time_per_init = min(time_per_init, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(inits[i], time_limit=time_per_init)
        update_best(refined)
        
    # If time remains, try a global search from the best solution
    if best_bary is not None and time.time() < deadline - 20:
        try:
            # Use Differential Evolution to explore the space, starting near best
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -calc_min_area(bary)
            
            bounds_de = [(0, 1)] * (N * 2)
            # Initialize DE with points near the best solution to guide it?
            # DE doesn't easily take a single seed point, but we can use 'sobol' or 'latinhypercube'
            # and hope it finds the basin. Or we can just run it.
            
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=50, # Keep it short
                tol=1e-12,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            
            if result_de.fun < -best_val:
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    # Refine the DE result
                    if time.time() < deadline - 8:
                        refined_de = refine_local(bary_de, time_limit=min(8.0, deadline - time.time() - 4))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final polish on best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def nm_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -calc_min_area(bary)
                
            result_nm = minimize(
                nm_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result_nm.fun < -best_val:
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
            
        # Final perturbation
        if best_bary is not None and time.time() < deadline - 1:
            rng_final = np.random.default_rng(999)
            step = 0.01
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                improved = False
                for _ in range(10):
                    pert = rng_final.normal(0, step, (N, 2))
                    new_bary = clamp_bary(best_bary + pert)
                    if update_best(new_bary):
                        improved = True
                        break
                if not improved:
                    step *= 0.7
                if step < 1e-7:
                    break

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
