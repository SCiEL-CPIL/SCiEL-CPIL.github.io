import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Define the topology: which points form triangles
# We'll use a structured layout:
# Row 0 (top): 1 point
# Row 1: 2 points  
# Row 2: 3 points
# Row 3: 3 points
# Row 4 (bottom): 2 points
# Total: 1+2+3+3+2 = 11

# Point indices:
# 0: top
# 1,2: row 1
# 3,4,5: row 2
# 6,7,8: row 3
# 9,10: row 4 (bottom)

# Triangles in this configuration (each triangle is 3 points)
tri_indices = [
    (0, 1, 2),
    (1, 3, 4), (1, 4, 2),
    (2, 4, 5),
    (3, 6, 7), (3, 7, 4),
    (4, 7, 8), (4, 8, 5),
    (5, 8, 10),
    (6, 7, 8),
    (7, 8, 9), (7, 9, 10),
    (8, 9, 10),
    (1, 3, 7), (1, 7, 2),  # Additional triangles for better coverage
    (2, 7, 8), (2, 8, 5),
    (3, 6, 9), (3, 9, 4),
    (4, 9, 10), (4, 10, 5),
    (6, 9, 10),
]
tri_idx = np.array(tri_indices)

# Triangle vertices
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
    """Convert Cartesian coords to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - 0.5 * (1 - y / H)
    w = y / H
    u = 1 - v - w
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

def edge_vector_obj(params):
    """
    Objective function using edge vector parametrization.
    
    params: flattened array of edge vectors
    We define the geometry using a subset of edge vectors that determine
    the positions of all 11 points.
    
    Strategy: Use the triangular lattice structure.
    - Point 0 is at the top vertex
    - Define edges from point 0 to points 1, 2
    - Define edges from points 1, 2 to points 3, 4, 5
    - Continue down the lattice
    
    Parameters:
    - e01, e02: vectors from point 0 to points 1, 2 (4 params)
    - e13, e14, e24, e25: vectors from points 1,2 to next row (8 params)
    - e36, e37, e47, e48, e58: vectors to next row (10 params)
    - e69, e79, e710, e89, e810: vectors to bottom row (10 params)
    
    Total: ~32 parameters vs 22 for direct coordinates
    """
    # Unpack edge vectors
    # Point 0 position (fixed at top for symmetry)
    p0 = np.array([0.5, H])
    
    # Row 1: points 1, 2
    e01 = params[0:2]
    e02 = params[2:4]
    p1 = p0 + e01
    p2 = p0 + e02
    
    # Row 2: points 3, 4, 5
    e13 = params[4:6]
    e14 = params[6:8]
    e24 = params[8:10]
    e25 = params[10:12]
    p3 = p1 + e13
    p4 = p1 + e14  # Also = p2 + e24 (consistency check)
    p5 = p2 + e25
    
    # Row 3: points 6, 7, 8
    e36 = params[12:14]
    e37 = params[14:16]
    e47 = params[16:18]
    e48 = params[18:20]
    e58 = params[20:22]
    p6 = p3 + e36
    p7 = p3 + e37  # Also = p4 + e47
    p8 = p4 + e48  # Also = p5 + e58
    
    # Row 4: points 9, 10
    e69 = params[22:24]
    e79 = params[24:26]
    e710 = params[26:28]
    e89 = params[28:30]
    e810 = params[30:32]
    p9 = p6 + e69  # Also = p7 + e79
    p10 = p7 + e710  # Also = p8 + e810
    
    # Assemble all points
    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    
    # Convert to barycentric
    bary = cart_to_bary(points)
    bary = clamp_bary(bary)
    
    # Calculate areas
    areas = calc_areas_bary(bary)
    
    return -np.min(areas)

def soft_edge_obj(params, beta=50.0):
    """Smooth version of edge vector objective."""
    p0 = np.array([0.5, H])
    
    e01 = params[0:2]
    e02 = params[2:4]
    p1 = p0 + e01
    p2 = p0 + e02
    
    e13 = params[4:6]
    e14 = params[6:8]
    e24 = params[8:10]
    e25 = params[10:12]
    p3 = p1 + e13
    p4 = p1 + e14
    p5 = p2 + e25
    
    e36 = params[12:14]
    e37 = params[14:16]
    e47 = params[16:18]
    e48 = params[18:20]
    e58 = params[20:22]
    p6 = p3 + e36
    p7 = p3 + e37
    p8 = p4 + e48
    
    e69 = params[22:24]
    e79 = params[24:26]
    e710 = params[26:28]
    e89 = params[28:30]
    e810 = params[30:32]
    p9 = p6 + e69
    p10 = p7 + e710
    
    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    bary = cart_to_bary(points)
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

def generate_lattice_init():
    """Generate initial configuration based on regular triangular lattice."""
    # Regular lattice spacing
    s = 0.25
    
    # Point 0: top
    p0 = np.array([0.5, H])
    
    # Row 1
    p1 = np.array([0.5 - s, H - s * np.sqrt(3) / 2])
    p2 = np.array([0.5 + s, H - s * np.sqrt(3) / 2])
    
    # Row 2
    p3 = np.array([0.5 - 2*s, H - 2*s * np.sqrt(3) / 2])
    p4 = np.array([0.5, H - 2*s * np.sqrt(3) / 2])
    p5 = np.array([0.5 + 2*s, H - 2*s * np.sqrt(3) / 2])
    
    # Row 3
    p6 = np.array([0.5 - 1.5*s, H - 3*s * np.sqrt(3) / 2])
    p7 = np.array([0.5 - 0.5*s, H - 3*s * np.sqrt(3) / 2])
    p8 = np.array([0.5 + 0.5*s, H - 3*s * np.sqrt(3) / 2])
    
    # Row 4
    p9 = np.array([0.5 - s, H - 4*s * np.sqrt(3) / 2])
    p10 = np.array([0.5 + s, H - 4*s * np.sqrt(3) / 2])
    
    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    bary = cart_to_bary(points)
    bary = clamp_bary(bary)
    
    # Extract edge vectors
    params = np.zeros(32)
    params[0:2] = p1 - p0
    params[2:4] = p2 - p0
    
    params[4:6] = p3 - p1
    params[6:8] = p4 - p1
    params[8:10] = p4 - p2
    params[10:12] = p5 - p2
    
    params[12:14] = p6 - p3
    params[14:16] = p7 - p3
    params[16:18] = p7 - p4
    params[18:20] = p8 - p4
    params[20:22] = p8 - p5
    
    params[22:24] = p9 - p6
    params[24:26] = p9 - p7
    params[26:28] = p10 - p7
    params[28:30] = p9 - p8
    params[30:32] = p10 - p8
    
    return params

def generate_perturbed_init(base_params, sigma=0.05, seed=None):
    """Generate perturbed edge vectors."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_params.shape)
    return base_params + noise

def refine_local(params_init, time_limit=5.0):
    """Refine edge vector configuration."""
    start_time = time.time()
    
    params_current = params_init.copy()
    
    # Stage 1: Soft min with L-BFGS-B
    try:
        result1 = minimize(
            soft_edge_obj,
            params_current,
            args=(50.0,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < edge_vector_obj(params_current):
            params_current = result1.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return params_current
        
    # Stage 2: Hard min with Nelder-Mead
    try:
        result2 = minimize(
            edge_vector_obj,
            params_current,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        params_current = result2.x
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return params_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            edge_vector_obj,
            params_current,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        params_current = result3.x
    except Exception:
        pass
        
    return params_current

def params_to_bary(params):
    """Convert edge vector params to barycentric coordinates."""
    p0 = np.array([0.5, H])
    
    e01 = params[0:2]
    e02 = params[2:4]
    p1 = p0 + e01
    p2 = p0 + e02
    
    e13 = params[4:6]
    e14 = params[6:8]
    e24 = params[8:10]
    e25 = params[10:12]
    p3 = p1 + e13
    p4 = p1 + e14
    p5 = p2 + e25
    
    e36 = params[12:14]
    e37 = params[14:16]
    e47 = params[16:18]
    e48 = params[18:20]
    e58 = params[20:22]
    p6 = p3 + e36
    p7 = p3 + e37
    p8 = p4 + e48
    
    e69 = params[22:24]
    e79 = params[24:26]
    e710 = params[26:28]
    e89 = params[28:30]
    e810 = params[30:32]
    p9 = p6 + e69
    p10 = p7 + e710
    
    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    bary = cart_to_bary(points)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_params = None
    best_val = -np.inf
    
    def update_best(params):
        nonlocal best_params, best_val
        bary = params_to_bary(params)
        val = -edge_vector_obj(params)
        if val > best_val:
            best_val = val
            best_params = params.copy()
            return True
        return False

    # 1. Start with regular lattice
    lattice_init = generate_lattice_init()
    refined = refine_local(lattice_init, time_limit=8.0)
    update_best(refined)
    
    # 2. Multi-start optimization
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        if restart_count % 3 == 1:
            if best_params is not None:
                init = generate_perturbed_init(best_params, sigma=0.03, seed=2000 + restart_count)
            else:
                init = generate_lattice_init() + rng.normal(0, 0.1, 32)
        elif restart_count % 3 == 2:
            # Different lattice scale
            s = 0.15 + 0.05 * (restart_count % 5)
            p0 = np.array([0.5, H])
            p1 = np.array([0.5 - s, H - s * np.sqrt(3) / 2])
            p2 = np.array([0.5 + s, H - s * np.sqrt(3) / 2])
            p3 = np.array([0.5 - 2*s, H - 2*s * np.sqrt(3) / 2])
            p4 = np.array([0.5, H - 2*s * np.sqrt(3) / 2])
            p5 = np.array([0.5 + 2*s, H - 2*s * np.sqrt(3) / 2])
            p6 = np.array([0.5 - 1.5*s, H - 3*s * np.sqrt(3) / 2])
            p7 = np.array([0.5 - 0.5*s, H - 3*s * np.sqrt(3) / 2])
            p8 = np.array([0.5 + 0.5*s, H - 3*s * np.sqrt(3) / 2])
            p9 = np.array([0.5 - s, H - 4*s * np.sqrt(3) / 2])
            p10 = np.array([0.5 + s, H - 4*s * np.sqrt(3) / 2])
            points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
            bary = cart_to_bary(points)
            bary = clamp_bary(bary)
            # Reconstruct params from bary
            cart = bary_to_cart(bary)
            init = np.zeros(32)
            init[0:2] = cart[1] - cart[0]
            init[2:4] = cart[2] - cart[0]
            init[4:6] = cart[3] - cart[1]
            init[6:8] = cart[4] - cart[1]
            init[8:10] = cart[4] - cart[2]
            init[10:12] = cart[5] - cart[2]
            init[12:14] = cart[6] - cart[3]
            init[14:16] = cart[7] - cart[3]
            init[16:18] = cart[7] - cart[4]
            init[18:20] = cart[8] - cart[4]
            init[20:22] = cart[8] - cart[5]
            init[22:24] = cart[9] - cart[6]
            init[24:26] = cart[9] - cart[7]
            init[26:28] = cart[10] - cart[7]
            init[28:30] = cart[9] - cart[8]
            init[30:32] = cart[10] - cart[8]
            init += rng.normal(0, 0.05, 32)
        else:
            # Random perturbation of lattice
            init = generate_lattice_init() + rng.normal(0, 0.08, 32)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 3. Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            result_slsqp = minimize(
                edge_vector_obj,
                best_params,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(result_slsqp.x)
        except Exception:
            pass
        
        if best_params is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    edge_vector_obj,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(result_nm.x)
            except Exception:
            pass
        
        if best_params is not None and time.time() < deadline - 0.5:
            try:
                result_final = minimize(
                    edge_vector_obj,
                    best_params,
                    method='SLSQP',
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(result_final.x)
            except Exception:
            pass

    # Convert to cartesian
    best_bary = params_to_bary(best_params)
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
