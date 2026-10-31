import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

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
    # bary is (N, 2) where u, v are coords relative to V0, V1?
    # Standard barycentric: P = u*V0 + v*V1 + w*V2, u+v+w=1
    # If we use (u, v) as input, w = 1 - u - v
    # x = u*0 + v*1 + w*0.5 = v + 0.5*(1-u-v) = v + 0.5 - 0.5u - 0.5v = 0.5 - 0.5u + 0.5v
    # y = u*0 + v*0 + w*H = (1-u-v)*H = H - H*u - H*v
    # Let's stick to the inspiration's convention which seemed to work well.
    # Inspiration: x = v + 0.5*w, y = w*H. Here w = 1-u-v.
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
    
    # Area is proportional to the determinant of the barycentric coordinates
    # det = | u1 v1 w1 |
    #       | u2 v2 w2 |
    #       | u3 v3 w3 |
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u, v >= 0 and u+v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-clip to be safe
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid log(0) or numerical issues
    if np.any(areas < 1e-15):
        return -1e-15
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    # Numerical stability check
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.flatten()
    bary = bary.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    This is a strong prior for Heilbronn triangles.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try different scales of the hexagonal grid
    for s in np.linspace(0.05, 0.5, 30):
        points = []
        # Generate points in a triangular grid
        for i in range(10):
            for j in range(10):
                # Triangular lattice coordinates in barycentric-like space
                # We map a square grid to the triangle
                # u = i*s, v = j*s*sqrt(3)/2 ? No, let's use standard triangular lattice vectors
                # In barycentric (u,v) with w=1-u-v:
                # Vertex 0: (1,0) -> u=1, v=0? No, my bary_to_cart uses:
                # u corresponds to weight of V0?
                # Let's check bary_to_cart:
                # x = v + 0.5*w. If u=1, v=0, w=0 -> x=0. Correct (V0).
                # If u=0, v=1, w=0 -> x=1. Correct (V1).
                # If u=0, v=0, w=1 -> x=0.5. Correct (V2).
                
                # So (u,v) are coordinates such that P = u*V0 + v*V1 + (1-u-v)*V2.
                
                # Hexagonal lattice points in the triangle:
                # We can parameterize by rows.
                # Row k has points with w = k * s.
                # u + v = 1 - k*s.
                # Points are distributed along this line.
                
                pass # The simple loop above was a bit loose. Let's use a more robust generator.

    # Robust Hexagonal Generator
    rng = np.random.default_rng(42)
    
    # Generate a grid of points in the triangle
    # We want roughly 11 points.
    # A 4x4 grid has 16 points. A 3x3 has 9.
    # Let's try to place points at a triangular lattice.
    
    candidates = []
    
    # Define lattice spacing
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        pts = []
        # Iterate over "rows" parallel to the base (constant w)
        # w = 1 - u - v
        # Let w = k * scale
        max_k = int(1.0 / scale)
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            # u + v = 1 - w_val
            # Distribute points along this segment
            # Number of points in this row?
            # For a triangular lattice, the number of points decreases as w increases.
            # Let's just generate a grid and filter.
            
            # Simpler approach: Generate a fine grid and pick the best 11?
            # Or just define specific lattice points.
            
            # Let's use a standard triangular lattice in Cartesian and convert.
            # But we are in Barycentric.
            
            # Points in row k (w = k*scale):
            # u can range from 0 to 1-w_val.
            # Let's place points at intervals of scale.
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Randomly select N points from the lattice to break symmetry?
            # Or just take the first N?
            # Taking first N might bias towards corners.
            # Let's try to center them.
            # For now, take the first N.
            bary = np.array(pts[:N])
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
                
    if best_bary is None:
        # Fallback to random
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
    Refine a configuration using two stages:
    1. L-BFGS-B on the smooth soft-min surrogate.
    2. Nelder-Mead on the hard minimum.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,), # beta
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

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
    refined_hex = refine_local(hex_init, time_limit=10.0)
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
            # Use a moderate popsize and maxiter to fit in time
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-6,
                polish=False
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
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
        
        # If we have improved significantly, we might want to stop random starts
        # and focus on polishing the best
        if best_val > 0.035: # Heuristic threshold
             # Switch to perturbation mode
             pass

    # 4. Final high-precision polish: SLSQP then Nelder-Mead
    if best_bary is not None and time.time() < deadline - 3:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
