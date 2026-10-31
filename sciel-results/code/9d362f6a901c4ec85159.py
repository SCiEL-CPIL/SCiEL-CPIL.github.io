import numpy as np
from scipy.optimize import minimize
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

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
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
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

def optimize_active_set(bary, active_indices, time_limit=2.0):
    """
    Maximize the minimum area of only the triangles in active_indices.
    This is a structural change: we ignore the other triangles during this local step,
    assuming they are sufficiently larger.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Pre-extract the specific triangle indices for the active set
    # active_indices is a list of indices into tri_idx
    active_tri_idx = tri_idx[active_indices]
    
    def active_obj(bary_flat):
        bary = bary_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        
        ui = u[active_tri_idx[:, 0]]
        vi = v[active_tri_idx[:, 0]]
        wi = w[active_tri_idx[:, 0]]
        
        uj = u[active_tri_idx[:, 1]]
        vj = v[active_tri_idx[:, 1]]
        wj = w[active_tri_idx[:, 1]]
        
        uk = u[active_tri_idx[:, 2]]
        vk = v[active_tri_idx[:, 2]]
        wk = w[active_tri_idx[:, 2]]
        
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        areas = np.abs(det)
        
        return -np.min(areas)
    
    try:
        result = minimize(
            active_obj,
            bary.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        return bary

def refine_sa(bary_init, time_limit=5.0, K_active=5):
    """
    Structural Change: Simulated Annealing with Active Set Local Optimization.
    
    1. Identify K smallest triangles.
    2. Optimize ONLY those K triangles using L-BFGS-B.
    3. Perturb the configuration.
    4. Accept if better or with probability exp(-delta/T).
    5. Cool down.
    """
    start_time = time.time()
    current_bary = bary_init.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    best_bary = current_bary.copy()
    best_val = current_val
    
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.95
    max_iter = 500
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Identify Active Set (K smallest triangles)
        areas = calc_areas_bary(current_bary)
        # Get indices of K smallest areas
        active_indices = np.argsort(areas)[:K_active]
        
        # 2. Local Optimization on Active Set
        optimized_bary = optimize_active_set(current_bary, active_indices, time_limit=0.5)
        
        # Evaluate the optimized bary using the FULL objective
        new_val = -hard_min_obj(optimized_bary.flatten())
        delta = new_val - current_val
        
        # 3. Acceptance Criterion
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_bary = optimized_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        
        # 4. Perturbation (Random Kick)
        # Only perturb if we accepted or to escape local minima
        if i % 5 == 0:
            noise = rng.normal(0, T * 0.1, current_bary.shape)
            perturbed_bary = clamp_bary(current_bary + noise)
            pert_val = -hard_min_obj(perturbed_bary.flatten())
            
            # Accept perturbation if better or with prob
            delta_p = pert_val - current_val
            if delta_p > 0 or rng.random() < np.exp(-delta_p / max(T, 1e-9)):
                current_bary = perturbed_bary
                current_val = pert_val
                if current_val > best_val:
                    best_val = current_val
                    best_bary = current_bary.copy()
        
        # 5. Cool Down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

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

    # 1. Try Hexagonal Init + SA Refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_sa(hex_init, time_limit=10.0, K_active=5)
    update_best(refined_hex)

    # 2. Multi-start with SA
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        
        # Alternate between random and perturbed best
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            noise = rng.normal(0, 0.05, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Use SA with varying K_active to explore different structures
        k_active = 5 if restart_count % 3 == 0 else 4
        refined = refine_sa(init, time_limit=4.0, K_active=k_active)
        update_best(refined)
        
        # Occasionally try a higher K to smooth out
        if restart_count % 5 == 0:
            refined_k = refine_sa(init, time_limit=3.0, K_active=8)
            update_best(refined_k)

    # 3. Final Polish: Use full objective with high precision
    if best_bary is not None and time.time() < deadline - 5:
        # Use a final SA run with very low T and high precision active set optimization
        # Or just a final L-BFGS-B on the full objective
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One last SA polish with very small steps
        if best_bary is not None and time.time() < deadline - 2:
            final_sa = refine_sa(best_bary, time_limit=2.0, K_active=5)
            update_best(final_sa)

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
