import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

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

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def calc_soft_min_area(points, beta=100.0):
    """
    Calculate a smooth approximation of the minimum area using log-sum-exp.
    This creates a differentiable objective for gradient-based methods.
    """
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # As beta -> inf, this approaches the true min
    max_area = np.max(areas)
    shifted = areas - max_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = max_area + log_sum / beta
    
    return soft_min

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

def hard_min_obj_cart(cart_flat):
    """Objective using hard min in Cartesian coords."""
    cart = cart_flat.reshape(N, 2)
    return -calc_min_area_cartesian(cart)

def soft_min_obj_cart(cart_flat, beta=100.0):
    """Objective using soft min in Cartesian coords."""
    cart = cart_flat.reshape(N, 2)
    return -calc_soft_min_area(cart, beta)

def project_to_triangle(cart):
    """Project points to be inside or on the boundary of the triangle."""
    bary = cart_to_bary(cart)
    bary = clamp_bary(bary, eps=1e-12)
    return bary_to_cart(bary)

def generate_lattice_init():
    """Generate a triangular lattice initialization."""
    # Side-4 triangular lattice gives 15 points, we need 11
    # Use a subset: corners + edge midpoints + some interior
    pts = []
    
    # 3 corners
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, H])
    
    # 3 edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.75, H/4])
    pts.append([0.25, H/4])
    
    # 5 interior points from a smaller lattice
    # Center and 4 around it
    cx, cy = 1/3, H/3
    r = 0.1
    pts.append([cx, cy])
    for angle in [0, 2*np.pi/5, 4*np.pi/5, 6*np.pi/5, 8*np.pi/5]:
        pts.append([cx + r*np.cos(angle), cy + r*np.sin(angle)])
    
    pts = np.array(pts)
    return project_to_triangle(pts)

def generate_hexagonal_init():
    """Generate a hexagonal-like arrangement."""
    pts = []
    
    # 3 corners
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, H])
    
    # 3 edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.75, H/4])
    pts.append([0.25, H/4])
    
    # Center
    cx, cy = 1/3, H/3
    pts.append([cx, cy])
    
    # 4 more points in a ring around center
    r = 0.15
    for angle in [np.pi/6, np.pi/6 + np.pi/2, np.pi/6 + np.pi, np.pi/6 + 3*np.pi/2]:
        pts.append([cx + r*np.cos(angle), cy + r*np.sin(angle)])
    
    pts = np.array(pts)
    return project_to_triangle(pts)

def generate_random_init(seed=None):
    """Generate random valid Cartesian coordinates inside the triangle."""
    rng = np.random.default_rng(seed)
    bary_u = rng.random(N)
    bary_v = rng.random(N) * (1 - bary_u)
    bary = np.column_stack([bary_u, bary_v])
    return bary_to_cart(bary)

def refine_local(cart_init, time_limit=5.0):
    """
    Refine a configuration using multi-stage local optimization.
    Stage 1: Soft-min with BFGS (smooth, fast convergence)
    Stage 2: Hard-min with Nelder-Mead (handles non-smoothness)
    Stage 3: Hard-min with SLSQP (polish)
    """
    start_time = time.time()
    cart_current = cart_init.copy()
    
    # Stage 1: Soft-min optimization with BFGS
    try:
        def obj_soft(x):
            cart = project_to_triangle(x.reshape(N, 2))
            return -calc_soft_min_area(cart, beta=50.0)
        
        result1 = minimize(
            obj_soft,
            cart_current.flatten(),
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-10}
        )
        cart_current = project_to_triangle(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return cart_current
    
    # Stage 2: Nelder-Mead on hard min
    try:
        def obj_hard(x):
            cart = project_to_triangle(x.reshape(N, 2))
            return -calc_min_area_cartesian(cart)
        
        result2 = minimize(
            obj_hard,
            cart_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        cart_current = project_to_triangle(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return cart_current
    
    # Stage 3: SLSQP polish on hard min
    try:
        def obj_slsqp(x):
            cart = project_to_triangle(x.reshape(N, 2))
            return -calc_min_area_cartesian(cart)
        
        # Simple bounds: x in [0, 1], y in [0, H]
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, H))
        
        result3 = minimize(
            obj_slsqp,
            cart_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        cart_current = project_to_triangle(result3.x.reshape(N, 2))
    except Exception:
        pass
    
    return cart_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_cart = None
    best_val = -np.inf
    
    def update_best(cart):
        nonlocal best_cart, best_val
        cart = project_to_triangle(cart)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_cart = cart.copy()
            return True
        return False

    # 1. Try structured initializations
    for init_gen in [generate_lattice_init, generate_hexagonal_init]:
        if time.time() > deadline - 20:
            break
        init = init_gen()
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)
    
    # 2. Global search with Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                cart = project_to_triangle(x.reshape(N, 2))
                return -calc_min_area_cartesian(cart)
            
            # Bounds: x in [0, 1], y in [0, H]
            bounds = []
            for i in range(N):
                bounds.append((0, 1))
                bounds.append((0, H))
            
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=15,
                popsize=12,
                tol=1e-4,
                polish=False,
                mutation=(0.5, 1.5),
                recombination=0.8
            )
            
            de_cart = project_to_triangle(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_cart, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random and perturbed restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        
        if best_cart is not None and restart_count % 3 == 0:
            # Perturb best solution
            noise = rng.normal(0, 0.04, best_cart.shape)
            init = project_to_triangle(best_cart + noise)
        else:
            # Random initialization
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 12 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(4.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
    
    # 4. Final high-precision polish on the best candidate
    if best_cart is not None and time.time() < deadline - 5:
        try:
            # Multiple rounds of Nelder-Mead with decreasing step size
            for _ in range(2):
                if time.time() > deadline - 3:
                    break
                result_nm = minimize(
                    lambda x: -calc_min_area_cartesian(project_to_triangle(x.reshape(N, 2))),
                    best_cart.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(project_to_triangle(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        # Final SLSQP polish
        if best_cart is not None and time.time() < deadline - 2:
            try:
                bounds = []
                for i in range(N):
                    bounds.append((0, 1))
                    bounds.append((0, H))
                
                result_slsqp = minimize(
                    lambda x: -calc_min_area_cartesian(project_to_triangle(x.reshape(N, 2))),
                    best_cart.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(project_to_triangle(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    if best_cart is None:
        # Fallback: return lattice initialization
        best_cart = generate_lattice_init()
    
    return best_cart
