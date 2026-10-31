import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    bary: (N, 2) array.
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= 0, v >= 0, u+v <= 1
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    return np.column_stack([u, v])

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            areas = calc_areas_bary(bary)
            val = np.min(areas)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def run_slsqp(bary_init):
        """
        Run SLSQP optimization on the epigraph formulation.
        Variables: [u1, v1, ..., uN, vN, t]
        """
        if time.time() > deadline - 5:
            return
        
        # Initial t can be the min area of the init
        init_areas = calc_areas_bary(bary_init)
        t_init = np.min(init_areas)
        
        x0 = np.concatenate([bary_init.flatten(), [t_init]])
        
        # Objective: minimize -t
        def objective(x):
            t = x[-1]
            bary = x[:-1].reshape(N, 2)
            return -t
        
        # Constraints: t - Area_i <= 0  => Area_i - t >= 0
        # scipy minimize expects constraints as func(x) >= 0 for 'ineq'
        def constraints_func(x):
            t = x[-1]
            bary = x[:-1].reshape(N, 2)
            areas = calc_areas_bary(bary)
            return areas - t
        
        # Bounds for barycentric coords
        # u_i in [0, 1], v_i in [0, 1]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # u_i
            bounds.append((0, 1)) # v_i
        bounds.append((0, 1)) # t
        
        # Additional inequality constraints: u_i + v_i <= 1
        # This is linear: 1 - u_i - v_i >= 0
        def triangle_constraint_func(x):
            bary = x[:-1].reshape(N, 2)
            u = bary[:, 0]
            v = bary[:, 1]
            return 1.0 - (u + v)
        
        cons = [
            {'type': 'ineq', 'fun': constraints_func},
            {'type': 'ineq', 'fun': triangle_constraint_func}
        ]
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                res_bary = res.x[:-1].reshape(N, 2)
                update_best(res_bary)
                
        except Exception:
            pass

    def run_soft_min_polish(bary_init, beta=500.0, maxiter=1000):
        """
        Polish using soft-min objective.
        """
        if time.time() > deadline - 5:
            return
        
        def soft_min_obj(bary_flat):
            bary = bary_flat.reshape(N, 2)
            bary = clamp_bary(bary)
            areas = calc_areas_bary(bary)
            
            # Soft min: -1/beta * log(sum(exp(-beta * areas)))
            scaled = -beta * areas
            max_scaled = np.max(scaled)
            if max_scaled > 700:
                return -np.min(areas)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min = -log_sum / beta
            return -soft_min

        bounds = [(0, 1)] * (N * 2)
        
        try:
            res = minimize(
                soft_min_obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            update_best(res.x.reshape(N, 2))
            
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    run_slsqp(hex_init)
    
    # 2. Try multiple random initializations
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        run_slsqp(rand_init)
        
    # 3. Polish the best found with soft-min
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            # Slight perturbation to escape potential flat spots
            rng = np.random.default_rng(200 + i)
            perturbed = best_bary + rng.normal(0, 0.01, best_bary.shape)
            perturbed = clamp_bary(perturbed)
            run_soft_min_polish(perturbed, beta=500.0, maxiter=200)
            
            # Also try SLSQP on the perturbed best
            run_slsqp(perturbed)

    # 4. Final high-precision polish on the absolute best
    if best_bary is not None and time.time() < deadline - 5:
        run_soft_min_polish(best_bary, beta=1000.0, maxiter=500)

    # 5. Convert best barycentric to Cartesian and return
    if best_bary is not None:
        return bary_to_cart(best_bary)
    else:
        # Fallback: return random valid points
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        return bary_to_cart(bary)
