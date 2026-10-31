import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def smooth_min_obj(bary_flat, beta=100.0):
    """Smooth surrogate for the hard min via log-sum-exp."""
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

def generate_hexagonal_init(scale=0.25):
    """Generate a hexagonal grid based initial configuration."""
    pts = []
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w_val = k * scale
        if w_val > 1.0:
            break
        num_pts_in_row = int((1 - w_val) / scale) + 1
        for j in range(num_pts_in_row):
            u_val = j * scale
            v_val = (1 - w_val) - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    if len(pts) >= N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N, replace=False)
        return np.array(pts)[idx].flatten()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None:
            return
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    # Bounds for DE: each barycentric coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Phase 1: Differential Evolution global search
    # Use a moderate population and max iterations to stay within time
    # DE with smooth objective (beta=50) for smoother landscape
    def de_obj(x):
        return smooth_min_obj(x, beta=50.0)
    
    try:
        # Use seed for reproducibility
        de_result = differential_evolution(
            de_obj,
            bounds=bounds,
            maxiter=80,
            popsize=20,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,  # We'll do our own polish
            init='sobol',
            updating='immediate',
            callback=lambda x, convergence: None
        )
        
        if de_result.success and de_result.x is not None:
            de_bary = de_result.x.reshape(N, 2)
            de_bary = clamp_bary(de_bary)
            update_best(de_bary)
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for refinement, return what we have
        if best_bary is not None:
            cart = bary_to_cart(best_bary)
            return cart
        # Fallback
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        cart = bary_to_cart(best_bary)
        return cart
    
    # Phase 2: Local refinement from multiple starting points
    # Collect several starting points: DE result + hexagonal + random
    start_points = []
    
    if best_bary is not None:
        start_points.append(best_bary.flatten())
    
    hex_init = generate_hexagonal_init(0.25)
    if hex_init is not None:
        start_points.append(hex_init)
    
    # Add a few random starts
    rng = np.random.default_rng(123)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        start_points.append(np.column_stack([u, v]).flatten())
    
    # Refine each start point
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        
        try:
            # Nelder-Mead on hard objective
            nm_result = minimize(
                hard_min_obj,
                sp,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if nm_result.x is not None:
                nm_bary = clamp_bary(nm_result.x.reshape(N, 2))
                update_best(nm_bary)
        except Exception:
            pass
        
        # SLSQP polish on the best found so far
        if time.time() > deadline - 5:
            break
        
        if best_bary is not None:
            try:
                slsqp_result = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-15
                    }
                )
                if slsqp_result.x is not None:
                    slsqp_bary = clamp_bary(slsqp_result.x.reshape(N, 2))
                    update_best(slsqp_bary)
            except Exception:
                pass
    
    # Phase 3: Final high-precision refinement
    if best_bary is not None and time.time() < deadline - 3:
        try:
            # Very tight Nelder-Mead
            final_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if final_nm.x is not None:
                final_bary = clamp_bary(final_nm.x.reshape(N, 2))
                update_best(final_bary)
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    cart = bary_to_cart(best_bary)
    return cart
