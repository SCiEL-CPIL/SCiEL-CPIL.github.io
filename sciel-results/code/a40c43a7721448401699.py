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

def get_min_area(bary):
    """Get the minimum area for a given barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def simulated_annealing(start_bary, t_start, t_end, steps, rng):
    """
    Run Simulated Annealing starting from start_bary.
    Returns the best configuration found.
    """
    current_bary = start_bary.copy()
    current_min_area = get_min_area(current_bary)
    
    best_bary = current_bary.copy()
    best_min_area = current_min_area
    
    # Temperature schedule: linear decay
    temp_schedule = np.linspace(t_start, t_end, steps)
    
    # Step size for perturbations
    # We perturb in barycentric space.
    # A reasonable step size is around 0.01 to 0.05.
    step_scale = 0.05
    
    for i in range(steps):
        if i % 1000 == 0:
            # Check time
            pass # We rely on the outer loop for time checking to avoid overhead in SA inner loop
            
        T = temp_schedule[i]
        
        # Generate perturbation
        # Perturb all points slightly
        perturbation = rng.normal(0, step_scale, size=(N, 2))
        # Ensure perturbation doesn't immediately violate bounds too badly? 
        # clamp_bary handles it, but large jumps might be rejected.
        
        next_bary = current_bary + perturbation
        next_bary = clamp_bary(next_bary)
        
        next_min_area = get_min_area(next_bary)
        
        delta = next_min_area - current_min_area
        
        if delta > 0:
            # Improvement
            current_bary = next_bary
            current_min_area = next_min_area
            if current_min_area > best_min_area:
                best_min_area = current_min_area
                best_bary = current_bary.copy()
        else:
            # Worse, accept with probability exp(delta / T)
            # delta is negative
            if T > 1e-15:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_bary = next_bary
                    current_min_area = next_min_area
            # else: T is very small, reject all worsening moves (greedy hill climbing phase)
            
    return best_bary, best_min_area

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initialization."""
    pts = []
    scale = 0.3
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
        return np.array(pts)[idx]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initialization
    # Use multiple diverse starting points
    start_points = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        start_points.append(hex_init)
        
    # Random starts
    rng_master = np.random.default_rng(123)
    for _ in range(5):
        u = rng_master.random(N)
        v = rng_master.random(N) * (1 - u)
        start_points.append(np.column_stack([u, v]))
        
    # 2. Simulated Annealing Loop
    # We will run SA on each start point, or re-run with different parameters.
    # Given time constraints, we do a few long runs.
    
    # Parameters
    t_start = 0.5
    t_end = 1e-6
    steps = 5000  # Adjust based on time per step
    
    # Estimate time per step
    # get_min_area involves 165 determinants. Very fast in NumPy.
    # SA loop overhead is Python. 
    # Let's try to fit as many steps as possible.
    
    # We'll run SA for each start point, but monitor time.
    # To be safe, we'll run a fixed number of steps per start point, 
    # and if time remains, do more restarts or longer runs.
    
    # Strategy: 
    # 1. Quick SA on all 6 start points to get good candidates.
    # 2. If time remains, run longer SA on the best candidates.
    
    # Phase 1: Quick exploration
    quick_steps = 2000
    t_start_quick = 0.2
    t_end_quick = 1e-4
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        rng_sa = np.random.default_rng(hash(sp.tobytes()) % 2**32)
        # Use a deterministic seed based on the start point for reproducibility in debugging, 
        # but random is better for diversity. Let's use random.
        rng_sa = np.random.default_rng()
        
        best_bary_sa, best_val_sa = simulated_annealing(sp, t_start_quick, t_end_quick, quick_steps, rng_sa)
        update_best(best_bary_sa)
        
    # Phase 2: Deep refinement of the best candidate found so far
    if best_bary is not None and time.time() < deadline - 5:
        # Run longer SA on the current best
        # Or run SA on slightly perturbed versions of the best
        deep_steps = 20000 # Much longer
        t_start_deep = 0.1
        t_end_deep = 1e-8
        
        # Do 3 deep runs with different perturbations of the best
        for i in range(3):
            if time.time() > deadline - 2:
                break
            
            # Perturb the best slightly to get a new start
            rng_pert = np.random.default_rng(1000 + i)
            pert = rng_pert.normal(0, 0.02, size=(N, 2))
            start_point = best_bary + pert
            start_point = clamp_bary(start_point)
            
            rng_sa = np.random.default_rng(2000 + i)
            best_bary_sa, best_val_sa = simulated_annealing(start_point, t_start_deep, t_end_deep, deep_steps, rng_sa)
            update_best(best_bary_sa)
            
    # 3. Final Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
