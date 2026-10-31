import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def get_areas_cartesian(cart):
    """Compute all triangle areas."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def soft_min_obj_bary(bary_flat, beta=100.0):
    """
    Smooth approximation of the min-area objective using log-sum-exp.
    Returns negative soft-min area (for minimization).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    areas = get_areas_cartesian(cart)
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability:
    a_min = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - a_min))))
    soft_min = a_min - (1.0 / beta) * log_sum
    
    return -soft_min

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def active_triangles(cart, threshold_frac=0.05, max_count=20):
    """
    Find triangles with areas near the minimum (active constraints).
    Returns list of (i, j, k) indices.
    """
    areas = get_areas_cartesian(cart)
    min_area = np.min(areas)
    if min_area <= 0:
        return [(0, 1, 2)]
    
    threshold = min_area * (1 + threshold_frac)
    active = np.where(areas <= threshold)[0]
    
    if len(active) > max_count:
        # Take the smallest ones
        order = np.argsort(areas[active])
        active = active[order[:max_count]]
    
    return [(tri_idx[t, 0], tri_idx[t, 1], tri_idx[t, 2]) for t in active]

def coordinated_ascent(bary_init, time_limit=10.0, verbose=False):
    """
    Coordinated ascent: iteratively optimize individual points to increase
    the minimum area, focusing on active constraints.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_val = -np.inf
    
    def current_min():
        cart = bary_to_cart(bary)
        return calc_min_area_cartesian(cart)
    
    best_val = current_min()
    
    max_rounds = 50
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_limit - 2:
            break
        
        cart = bary_to_cart(bary)
        areas = get_areas_cartesian(cart)
        min_area = np.min(areas)
        
        # Get active triangles
        active = active_triangles(cart, threshold_frac=0.03, max_count=15)
        
        # For each point, try to move it to improve the minimum
        improved = False
        for pt_idx in range(N):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Collect constraints involving this point
            constraints = []
            for (i, j, k) in active:
                involved = set([i, j, k])
                if pt_idx in involved:
                    constraints.append((i, j, k))
            
            if not constraints:
                continue
            
            # Try small perturbations in barycentric space
            best_delta = None
            best_area = calc_min_area_cartesian(cart)
            
            # Random directions
            rng = np.random.default_rng(round_idx * 1000 + pt_idx)
            for attempt in range(30):
                # Random direction in 2D barycentric
                angle = rng.uniform(0, 2 * np.pi)
                magnitude = rng.uniform(0.01, 0.05)
                du = magnitude * np.cos(angle)
                dv = magnitude * np.sin(angle)
                
                new_bary = bary.copy()
                new_bary[pt_idx, 0] += du
                new_bary[pt_idx, 1] += dv
                new_bary = clamp_bary(new_bary)
                
                new_cart = bary_to_cart(new_bary)
                new_min = calc_min_area_cartesian(new_cart)
                
                if new_min > best_area + 1e-12:
                    best_area = new_min
                    best_delta = (du, dv)
            
            # Also try gradient-like direction: move away from centroid of active triangles
            if best_delta is None:
                # Compute average position of other points in active triangles
                other_pts = []
                for (i, j, k) in constraints:
                    for idx in [i, j, k]:
                        if idx != pt_idx:
                            other_pts.append(idx)
                
                if other_pts:
                    other_cart = cart[list(set(other_pts))]
                    centroid = np.mean(other_cart, axis=0)
                    my_cart = cart[pt_idx]
                    
                    # Direction away from centroid
                    direction = my_cart - centroid
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm
                        
                        # Convert cartesian direction to barycentric
                        # dx = dv + 0.5*(dw) where dw = -du - dv
                        # dy = H*dw
                        # So: dx = dv - 0.5*du - 0.5*dv = 0.5*dv - 0.5*du
                        #     dy = -H*du - H*dv
                        # Given (dx, dy), solve for (du, dv):
                        # dx = 0.5*dv - 0.5*du  =>  dv - du = 2*dx
                        # dy = -H*(du + dv)      =>  du + dv = -dy/H
                        # Solving: dv = dx - dy/(2H), du = -dy/(2H) - dx
                        dx_dir = direction[0]
                        dy_dir = direction[1]
                        du_est = -dy_dir / (2 * H) - dx_dir
                        dv_est = dx_dir - dy_dir / (2 * H)
                        
                        for scale in [0.02, 0.01, 0.005]:
                            new_bary = bary.copy()
                            new_bary[pt_idx, 0] += scale * du_est
                            new_bary[pt_idx, 1] += scale * dv_est
                            new_bary = clamp_bary(new_bary)
                            new_cart = bary_to_cart(new_bary)
                            new_min = calc_min_area_cartesian(new_cart)
                            if new_min > best_area + 1e-12:
                                best_area = new_min
                                best_delta = (scale * du_est, scale * dv_est)
                                break
            
            if best_delta is not None:
                bary[pt_idx, 0] += best_delta[0]
                bary[pt_idx, 1] += best_delta[1]
                bary = clamp_bary(bary)
                improved = True
        
        if not improved:
            break
    
    return bary, best_val

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
    s = 0.25
    hex_pts = []
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([s * np.cos(angle), s * np.sin(angle)])
    
    inner_pts = []
    inner_pts.append([0.0, 0.0])
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        inner_pts.append([0.15 * np.cos(angle), 0.15 * np.sin(angle)])
    
    all_pts = np.array(hex_pts + inner_pts)
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
    Refine a configuration using coordinated ascent + soft-min polish.
    """
    start_time = time.time()
    
    # Stage 1: Coordinated ascent (targets active constraints directly)
    bary_ca, _ = coordinated_ascent(bary_init, time_limit=time_limit * 0.5)
    bary_current = bary_ca
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 2: Soft-min SLSQP polish (smooth objective, better gradients)
    try:
        bounds = [(0, 1)] * (N * 2)
        result2 = minimize(
            soft_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            args=(80.0,),
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: Hard-min Nelder-Mead final polish
    try:
        result3 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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
    if time.time() < deadline - 50:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=8.0)
        update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
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
                popsize=10,
                tol=1e-4,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random restarts with coordinated ascent
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            scale = 0.03 + 0.01 * (restart_count // 10)
            init = generate_perturbed_init(best_bary, scale=scale, seed=5000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            scale = 0.06 + 0.01 * (restart_count // 10)
            init = generate_perturbed_init(best_bary, scale=scale, seed=6000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 20 - time.time()
        if remaining < 4:
            break
            
        t_limit = min(8.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_polish, _ = coordinated_ascent(best_bary, time_limit=5.0)
            update_best(bary_polish)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_slsqp = minimize(
                    soft_min_obj_bary,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    args=(150.0,),
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj_bary,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
