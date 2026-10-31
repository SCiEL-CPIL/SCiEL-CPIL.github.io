import numpy as np
from scipy.optimize import differential_evolution, minimize, L_BFGS_B
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def get_areas_cartesian(cart):
    """Compute all triangle areas."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    areas = get_areas_cartesian(points)
    return np.min(areas)

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

def soft_min_obj_bary(bary_flat, beta=50.0):
    """
    Smooth approximation of the min-area objective using log-sum-exp.
    Returns negative soft-min area (for minimization).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    areas = get_areas_cartesian(cart)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize min(areas), so we minimize -min(areas).
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # So we minimize 1/beta * log(sum(exp(-beta * areas)))
    
    # Shift for numerical stability
    min_a = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_a))
    soft_min_val = min_a - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min_val

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def generate_lattice_init():
    """Generate a triangular lattice initialization with 10 points + 1 center."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

def generate_hexagonal_init():
    """Generate a hexagonal packing-inspired initialization."""
    # 6 vertices of a regular hexagon near the boundary + 5 interior points
    # Scale to fit within the equilateral triangle
    s = 0.25  # scale factor
    hex_pts = []
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([s * np.cos(angle), s * np.sin(angle)])
    
    # Add 5 interior points
    inner_pts = []
    # Center
    inner_pts.append([0.0, 0.0])
    # 4 more points
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        inner_pts.append([0.15 * np.cos(angle), 0.15 * np.sin(angle)])
    
    all_pts = np.array(hex_pts + inner_pts)
    # Convert to barycentric and clamp
    bary = cart_to_bary(all_pts)
    bary = clamp_bary(bary)
    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, scale=0.05, seed=None):
    """Generate perturbed version of base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, scale, base.shape)
    perturbed = base + noise
    return clamp_bary(perturbed)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: L-BFGS-B on soft min (fast, smooth)
    try:
        result1 = minimize(
            soft_min_obj_bary,
            bary_current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 2: Nelder-Mead on hard min (polish)
    try:
        result2 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
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
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=10.0)
    update_best(refined_lat)
    
    # 1b. Hexagonal Init
    if time.time() < deadline - 60:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=8.0)
        update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 30:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                cart = bary_to_cart(bary)
                return -calc_min_area_cartesian(cart)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20,
                popsize=15,
                tol=1e-4,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        # Mix of random and perturbed best
        if best_bary is not None and restart_count % 3 == 0:
            scale = 0.03 + 0.02 * (restart_count // 10)
            init = generate_perturbed_init(best_bary, scale=scale, seed=5000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            scale = 0.08 + 0.02 * (restart_count // 10)
            init = generate_perturbed_init(best_bary, scale=scale, seed=6000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 20 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(6.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # Use L-BFGS-B again with very tight tolerances on soft min
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                soft_min_obj_bary,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj_bary,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
