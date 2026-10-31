from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

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
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def get_min_area_and_active(points, tol=1e-6):
    """Return min area and indices of points in active (near-minimum) triangles."""
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    # Find triangles with area close to minimum
    active_mask = areas <= min_area + tol
    active_triangles = tri_idx[active_mask]
    
    # Get unique point indices involved
    if len(active_triangles) > 0:
        active_points = np.unique(active_triangles.flatten())
    else:
        active_points = np.arange(N)
    
    return min_area, active_points

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
    Soft-min objective using log-sum-exp approximation.
    Maximizes the soft minimum of all triangle areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft minimum: -log(sum(exp(-beta * areas))) / beta
    # This approximates -min(areas) as beta -> infinity
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min  # We want to maximize, so negate

def epigraph_obj(params):
    """
    Epigraph formulation: parameters = [bary_coords (N*2), t]
    where t is the minimum area.
    Objective: minimize -t subject to all triangle areas >= t
    Using soft constraints for differentiability.
    """
    bary = params[:N*2].reshape(N, 2)
    t = params[N*2]
    
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    
    # Penalize if any area < t, and maximize t
    # Use smooth penalty: sum of max(0, t - area)^2
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # Objective: -t + penalty
    return -t + penalty

def coordinated_ascent(bary_init, max_iters=100, time_limit=10.0):
    """
    Coordinated ascent: iteratively optimize subsets of points 
    involved in minimum-area triangles.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    best_val = calc_min_area_cartesian(bary_to_cart(bary_current))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        cart = bary_to_cart(bary_current)
        min_area, active_points = get_min_area_and_active(cart, tol=1e-5)
        
        # Optimize only active points using Nelder-Mead
        active_idx = list(active_points)
        if len(active_idx) < 2:
            active_idx = list(range(N))
        
        # Extract active point coordinates
        active_bary = bary_current[active_idx].copy()
        
        def sub_obj(x):
            new_bary = bary_current.copy()
            new_bary[active_idx] = x.reshape(len(active_idx), 2)
            new_bary = clamp_bary(new_bary)
            new_cart = bary_to_cart(new_bary)
            return -calc_min_area_cartesian(new_cart)
        
        try:
            result = minimize(
                sub_obj,
                active_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            new_active = clamp_bary(result.x.reshape(len(active_idx), 2))
            new_bary = bary_current.copy()
            new_bary[active_idx] = new_active
            new_bary = clamp_bary(new_bary)
            
            new_val = calc_min_area_cartesian(bary_to_cart(new_bary))
            
            if new_val > best_val:
                bary_current = new_bary
                best_val = new_val
            else:
                # If no improvement, try moving all points slightly
                break
        except Exception:
            break
    
    return bary_current, best_val

def generate_lattice_init():
    """Generate a triangular lattice initialization."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

def generate_heilbronn_structure():
    """
    Generate initialization based on known Heilbronn triangle structure.
    Uses symmetric placement for 11 points.
    """
    # Start with vertices and edge midpoints
    pts = [
        [0.0, 0.0],      # V0
        [1.0, 0.0],      # V1
        [0.5, H],        # V2
        [0.5, 0.0],      # Midpoint V0-V1
        [0.25, H/4],     # Midpoint V0-V2
        [0.75, H/4],     # Midpoint V1-V2
    ]
    
    # Add interior points in a hexagonal pattern
    cx, cy = 1/3, H/3
    r = 0.12
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    
    pts = np.array(pts)
    bary = cart_to_bary(pts)
    bary = clamp_bary(bary)
    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using local optimization with soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead on soft-min (smoother)
    try:
        result1 = minimize(
            soft_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
    
    # Stage 2: Coordinated ascent for precision
    try:
        bary_ca, _ = coordinated_ascent(bary_current, max_iters=50, time_limit=time_limit * 0.4)
        bary_current = bary_ca
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: Final SLSQP polish
    try:
        result2 = minimize(
            soft_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14}
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

    # 1. Heilbronn structure init
    heil_init = generate_heilbronn_structure()
    refined_heil = refine_local(heil_init, time_limit=12.0)
    update_best(refined_heil)
    
    # 2. Lattice Init
    if time.time() < deadline - 30:
        lat_init = generate_lattice_init()
        refined_lat = refine_local(lat_init, time_limit=8.0)
        update_best(refined_lat)
    
    # 3. Global Search with Differential Evolution (reduced iterations)
    if time.time() < deadline - 25:
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
                maxiter=25,
                popsize=12,
                tol=1e-4,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Extended coordinated ascent on best solution
    if time.time() < deadline - 5 and best_bary is not None:
        try:
            bary_final, _ = coordinated_ascent(best_bary, max_iters=200, time_limit=min(15.0, deadline - 5 - time.time()))
            update_best(bary_final)
        except Exception:
            pass
    
    # 5. Final refinement if time remains
    if time.time() < deadline - 2 and best_bary is not None:
        try:
            bary_polish = refine_local(best_bary, time_limit=min(5.0, deadline - 2 - time.time()))
            update_best(bary_polish)
        except Exception:
            pass
    
    cart = bary_to_cart(best_bary)
    return cart
