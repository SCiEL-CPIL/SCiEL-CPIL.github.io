import numpy as np
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def smooth_min_area(bary, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    We maximize this value.
    log-sum-exp(-areas) approximates -min(areas).
    So we want to minimize log-sum-exp(-areas).
    Or maximize -log-sum-exp(-areas).
    """
    areas = calc_areas_bary(bary)
    # To avoid overflow/underflow in exp, we can scale
    # min(areas) is roughly the target.
    # log(sum(exp(-beta * areas)))
    # Let m = min(areas). sum(exp(-beta*areas)) = exp(-beta*m) * sum(exp(-beta*(areas-m)))
    # log(...) = -beta*m + log(sum(exp(-beta*(areas-m))))
    # We want to maximize min(areas), which is equivalent to minimizing this quantity.
    m = np.min(areas)
    shifted = -beta * (areas - m)
    # If areas are all equal to m, shifted is 0, sum is 165, log is log(165).
    # If one area is slightly larger, its term becomes smaller.
    # Actually, we want a smooth function that is small when min area is large.
    # Let's use the soft-min:
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    # This is a smooth approximation of min(areas).
    # We want to MAXIMIZE soft_min(areas).
    
    # Numerical stability:
    # log(sum(exp(-beta * areas)))
    # Let max_exp_arg = max(-beta * areas) = -beta * min(areas) = -beta * m
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum(exp(-beta * areas))) = -beta * m + log(sum(exp(-beta * (areas - m))))
    
    # soft_min = -1/beta * [ -beta * m + log(sum(exp(-beta * (areas - m)))) ]
    # soft_min = m - 1/beta * log(sum(exp(-beta * (areas - m))))
    
    # Note: areas - m >= 0, so -beta * (areas - m) <= 0.
    # exp(-beta * (areas - m)) <= 1.
    # sum is between 1 and 165.
    # log(sum) is between 0 and log(165).
    # So soft_min is between m - log(165)/beta and m.
    # As beta -> inf, soft_min -> m.
    
    exp_sum = np.sum(np.exp(-beta * (areas - m)))
    if exp_sum < 1e-300:
        return m
    return m - (1.0 / beta) * np.log(exp_sum)

def objective(x, beta):
    """
    Objective function to MINIMIZE.
    We want to maximize smooth_min_area, so we minimize -smooth_min_area.
    """
    bary = x[:N*2].reshape(N, 2)
    # Clamp to ensure validity during optimization
    bary = clamp_bary(bary)
    val = smooth_min_area(bary, beta)
    return -val

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
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
                val = min_area(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
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

def nelder_mead_optimize(bary_init, beta, max_iter=200, time_limit=5.0):
    """
    Run Nelder-Mead optimization on the barycentric coordinates.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    x0 = bary_init.flatten()
    
    # Bounds are handled by clamping in objective, but NM doesn't support bounds directly.
    # We rely on the clamping to keep points inside.
    
    try:
        result = minimize(
            objective,
            x0,
            args=(beta,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if time.time() > start_time + time_limit:
            return bary_init, min_area(bary_init)
            
        best_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
        val = min_area(best_bary)
        return best_bary, val
        
    except Exception:
        return bary_init, min_area(bary_init)

def simulated_annealing(bary_init, time_limit=30.0):
    """
    Perform simulated annealing to find a better configuration.
    """
    start_time = time.time()
    current_bary = bary_init.copy()
    current_val = min_area(current_bary)
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.99
    
    rng = np.random.default_rng(42)
    
    while time.time() < start_time + time_limit:
        # Generate perturbation
        noise = rng.normal(0, T, current_bary.shape)
        candidate_bary = clamp_bary(current_bary + noise)
        candidate_val = min_area(candidate_bary)
        
        # Acceptance criterion
        delta = candidate_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_bary = candidate_bary
            current_val = candidate_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse initial configurations
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Random
    for i in range(5):
        inits.append(generate_random_init(seed=1000+i))
        
    # 2. Refine each initialization with Nelder-Mead and Simulated Annealing
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        init_val = min_area(init_bary)
        update_best(init_bary)
        
        # Phase 1: Nelder-Mead with increasing beta (sharpness)
        # Start with lower beta for smoother landscape, increase for precision
        current_bary = init_bary.copy()
        current_val = init_val
        
        for beta in [10.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            refined_bary, refined_val = nelder_mead_optimize(current_bary, beta, max_iter=100, time_limit=3.0)
            if refined_val > current_val:
                current_bary = refined_bary
                current_val = refined_val
            update_best(current_bary)
            
        # Phase 2: Simulated Annealing to escape local minima
        if time.time() < deadline - 5:
            sa_bary, sa_val = simulated_annealing(current_bary, time_limit=5.0)
            update_best(sa_bary)
            
            # Final polish after SA
            if sa_val > current_val:
                current_bary = sa_bary
                current_val = sa_val
                # Quick NM polish
                refined_bary, refined_val = nelder_mead_optimize(current_bary, 200.0, max_iter=50, time_limit=2.0)
                update_best(refined_bary)

    # 3. Global search using Simulated Annealing from scratch with higher T
    if time.time() < deadline - 10:
        # Start from a random point with high temperature to explore globally
        global_init = generate_random_init(seed=9999)
        # Use a longer SA run
        global_bary, global_val = simulated_annealing(global_init, time_limit=10.0)
        update_best(global_bary)
        
        # Polish the global result
        if global_val > best_val * 0.9: # If it's competitive
            refined_bary, refined_val = nelder_mead_optimize(global_bary, 200.0, max_iter=100, time_limit=5.0)
            update_best(refined_bary)

    # 4. Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 2:
        # Use very high beta for precise minimization of the smooth min
        final_bary, final_val = nelder_mead_optimize(best_bary, 500.0, max_iter=100, time_limit=3.0)
        update_best(final_bary)

    # Convert to cartesian
    if best_bary is None:
        best_bary = generate_random_init(42)
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
