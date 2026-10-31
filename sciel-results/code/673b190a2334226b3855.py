import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

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
    
    # Ensure u >= 0, v >= 0, u + v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_value(bary):
    """Compute the hard minimum area value."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def smooth_obj(bary_flat, p=20.0):
    """
    Smooth surrogate using log-sum-exp for soft min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    areas = np.maximum(areas, 1e-15)
    
    beta = p
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp / beta
    return -soft_min

def simulated_annealing(bary_init, time_limit=30.0, seed=42):
    """
    Simulated annealing on barycentric coordinates with adaptive step size.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    bary_current = bary_init.copy()
    val_current = hard_min_value(bary_current)
    
    bary_best = bary_current.copy()
    val_best = val_current
    
    T0 = 0.03
    T_end = 0.0005
    alpha = 0.997  # cooling rate
    
    T = T0
    step_size = 0.02
    
    # Track best step size
    best_step = step_size
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and T > T_end:
        # Generate perturbation
        noise = rng.normal(0, step_size, bary_current.shape)
        bary_candidate = clamp_bary(bary_current + noise)
        
        val_candidate = hard_min_value(bary_candidate)
        
        # Accept if better, or with probability exp((val_candidate - val_current) / T)
        if val_candidate >= val_current:
            bary_current = bary_candidate
            val_current = val_candidate
            no_improve_count = 0
            if val_candidate > val_best:
                bary_best = bary_candidate.copy()
                val_best = val_candidate
                best_step = step_size
        else:
            delta = val_candidate - val_current  # negative
            prob = np.exp(delta / T)
            if rng.random() < prob:
                bary_current = bary_candidate
                val_current = val_candidate
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Adaptive step size: reduce if no improvement for a while
        if no_improve_count > max_no_improve:
            step_size *= 0.95
            no_improve_count = 0
        
        T *= alpha
    
    return bary_best

def coordinate_descent(bary_init, time_limit=5.0):
    """
    Coordinate descent: optimize one point at a time.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    val_current = hard_min_value(bary_current)
    
    bounds = [(0, 1)] * 2
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            # Optimize point i while keeping others fixed
            others = bary_current.copy()
            others[i] = 0.0  # dummy
            
            def obj_i(x):
                bary_temp = others.copy()
                bary_temp[i] = x
                bary_temp = clamp_bary(bary_temp)
                return -hard_min_value(bary_temp)
            
            x0 = bary_current[i].copy()
            try:
                result = minimize(
                    obj_i,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                )
                bary_current[i] = clamp_bary(result.x.reshape(1, 2))[0]
                val_new = hard_min_value(bary_current)
                if val_new > val_current:
                    val_current = val_new
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return bary_current

def refine_local(bary_init, time_limit=5.0):
    """
    Refine using multiple methods.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: L-BFGS-B with smooth objective
    try:
        result1 = minimize(
            smooth_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or -result1.fun > hard_min_value(bary_current):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Coordinate descent
    bary_current = coordinate_descent(bary_current, time_limit=time_limit * 0.4)
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: Nelder-Mead on hard min
    try:
        def hard_obj(x):
            return -hard_min_value(x.reshape(N, 2))
        
        result2 = minimize(
            hard_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
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
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = hard_min_value(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration.
    """
    # Use a pattern based on known good configurations
    # 11 points: try to use symmetry
    pts = []
    
    # Center point
    pts.append([1/3, 1/3])
    
    # 6 points around center in hexagonal pattern
    r = 0.15
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        # Convert to barycentric
        x = 0.5 + r * np.cos(rad)
        y = H/3 + r * np.sin(rad)
        # Inverse barycentric transform
        u = (2*x - 1)/1 + (2*y)/H  # This is approximate
        # Better: solve directly
        # x = v + 0.5*(1-u-v) = 0.5 + v - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
        # y = (1-u-v)*H
        # So: u = 1 - 2*y/H - v
        # x = 0.5 + 0.5*v - 0.5*(1 - 2*y/H - v) = 0.5 + 0.5*v - 0.5 + y/H + 0.5*v = v + y/H
        # So v = x - y/H
        # u = 1 - 2*y/H - v = 1 - 2*y/H - x + y/H = 1 - x - y/H
        
        v_val = x - y/H
        u_val = 1 - x - y/H
        if u_val >= 0 and v_val >= 0 and u_val + v_val <= 1:
            pts.append([u_val, v_val])
    
    # Add remaining points
    while len(pts) < N:
        rng = np.random.default_rng(len(pts))
        u = rng.random()
        v = rng.random() * (1 - u)
        pts.append([u, v])
    
    bary = np.array(pts[:N])
    return clamp_bary(bary)

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try multiple initial configurations
    inits = [
        generate_hexagonal_init(),
        generate_symmetric_init(),
        generate_random_init(42)
    ]
    
    for init in inits:
        refined = refine_local(init, time_limit=5.0)
        update_best(refined)
    
    # 2. Simulated annealing from best
    if time.time() < deadline - 30:
        sa_bary = simulated_annealing(best_bary, time_limit=20.0, seed=42)
        update_best(sa_bary)
    
    # 3. Multi-start with simulated annealing + refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            noise = rng.normal(0, 0.04, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Simulated annealing
        sa_bary = simulated_annealing(init, time_limit=4.0, seed=1000 + restart_count)
        # Refine
        refined = refine_local(sa_bary, time_limit=2.0)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 3:
        # Coordinate descent
        bary_cd = coordinate_descent(best_bary, time_limit=3.0)
        update_best(bary_cd)
        
        if time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                def hard_obj(x):
                    return -hard_min_value(x.reshape(N, 2))
                
                result_final = minimize(
                    hard_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
