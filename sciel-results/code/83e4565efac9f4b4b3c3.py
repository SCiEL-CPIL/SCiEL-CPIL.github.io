import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
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

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def soft_min_obj_bary(bary_flat, beta=50.0):
    """
    Objective using soft min (log-sum-exp approximation) for smoother gradients.
    As beta -> infinity, this approaches the hard min.
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
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This is smoother than hard min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

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

def refine_local(bary_init, time_limit=5.0, use_soft_min=False):
    """
    Refine a configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead on hard min (robust to non-smoothness)
    try:
        obj_func = soft_min_obj_bary if use_soft_min else hard_min_obj_bary
        result1 = minimize(
            obj_func,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: SLSQP polish with hard min for precise convergence
    try:
        result2 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
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
    good_candidates = []
    
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

    def track_good(bary):
        nonlocal best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        # Keep if within 2% of best or if we have fewer than 5 candidates
        if len(good_candidates) < 5 or val > best_val * 0.98:
            good_candidates.append((val, bary.copy()))
            # Keep only top 5
            good_candidates.sort(key=lambda x: -x[0])
            del good_candidates[5:]

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=8.0)
    update_best(refined_lat)
    track_good(refined_lat)
    
    # 1b. Hexagonal Init
    if time.time() < deadline - 60:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=6.0)
        update_best(refined_hex)
        track_good(refined_hex)
    
    # 1c. Soft-min refinement of lattice init (different landscape)
    if time.time() < deadline - 55:
        refined_lat_soft = refine_local(lat_init, time_limit=6.0, use_soft_min=True)
        update_best(refined_lat_soft)
        track_good(refined_lat_soft)
    
    # 2. Structural Change: Basin Hopping
    # Basin Hopping combines global random perturbations with local minimization.
    # It is particularly effective for non-convex, non-smooth landscapes like this.
    # We use a soft-min objective for the global phase to guide the search smoothly,
    # and Nelder-Mead for the local phase to handle the non-smoothness of the min-area.
    
    if time.time() < deadline - 10:
        def bh_obj(x):
            # Use soft min for the global search to avoid getting stuck in local minima
            # of the hard min landscape due to its non-differentiability.
            return soft_min_obj_bary(x, beta=20.0)
        
        def bh_local(x):
            # Local minimizer: Nelder-Mead on the hard min objective
            # This ensures we land in the true local optimum of the hard min.
            bary = x.reshape(N, 2)
            bary = clamp_bary(bary)
            result = minimize(
                hard_min_obj_bary,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            return result.x
        
        # Run Basin Hopping from the best current candidate
        if best_bary is not None:
            try:
                bh_result = basinhopping(
                    bh_obj,
                    best_bary.flatten(),
                    minimizer_kwargs={'options': {'maxiter': 500}},
                    niter=20, # Number of global steps
                    stepsize=0.1, # Perturbation scale
                    minimizer=bh_local,
                    seed=42
                )
                bh_bary = clamp_bary(bh_result.x.reshape(N, 2))
                # Refine the basin hopping result with a longer local refinement
                refined_bh = refine_local(bh_bary, time_limit=10.0)
                update_best(refined_bh)
                track_good(refined_bh)
            except Exception:
                pass
    
    # 3. Multi-start: random restarts with local refinement
    # This incorporates the structural change from inspiration: mixing random and perturbed best
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
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
        
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(5.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
        track_good(refined)
    
    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj_bary,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass

    cart = bary_to_cart(best_bary)
    
    return cart
