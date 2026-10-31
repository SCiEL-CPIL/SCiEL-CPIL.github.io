import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def perimeter_to_bary(t):
    """
    Map a scalar t in [0, 1) to a point on the perimeter of the unit equilateral triangle.
    Perimeter is 3.
    Side 1: (0,0) to (1,0). Length 1.
    Side 2: (1,0) to (0.5, sqrt(3)/2). Length 1.
    Side 3: (0.5, sqrt(3)/2) to (0,0). Length 1.
    """
    t = np.mod(t, 1.0)
    bary = np.zeros((len(t), 2))
    
    # Side 1: u=0, v=t (if t < 1/3)
    mask1 = t < (1.0/3.0)
    bary[mask1, 0] = 0.0
    bary[mask1, 1] = t[mask1]
    
    # Side 2: v=1, u=t-1/3 (if 1/3 <= t < 2/3)
    mask2 = (t >= (1.0/3.0)) & (t < (2.0/3.0))
    bary[mask2, 0] = t[mask2] - (1.0/3.0)
    bary[mask2, 1] = 1.0 - bary[mask2, 0] # w=0 => u+v=1
    
    # Side 3: u=1, v=t-2/3? No.
    # Vertex V2 is (0.5, H). Bary (0,0) relative to V0,V1,V2? 
    # V0=(0,0) -> (1,0) in bary? No, standard bary:
    # P = u*V0 + v*V1 + w*V2.
    # V0=(0,0) => u=1, v=0, w=0.
    # V1=(1,0) => u=0, v=1, w=0.
    # V2=(0.5, H) => u=0, v=0, w=1.
    
    # Let's re-evaluate barycentric mapping carefully.
    # My bary_to_cart uses:
    # x = v + 0.5*w
    # y = w*H
    # V0 (0,0): u=1, v=0, w=0 -> x=0, y=0. Correct.
    # V1 (1,0): u=0, v=1, w=0 -> x=1, y=0. Correct.
    # V2 (0.5, H): u=0, v=0, w=1 -> x=0.5, y=H. Correct.
    
    # Side 1: V0 to V1. w=0. u+v=1.
    # P = (1-t)V0 + t V1.
    # Bary: u=1-t, v=t, w=0.
    # My previous mask1 had u=0, v=t. That is wrong for V0->V1.
    # Let's fix perimeter mapping.
    
    bary = np.zeros((len(t), 2))
    
    # Side 1: V0(1,0,0) to V1(0,1,0).
    # Param s in [0,1]. Bary: u=1-s, v=s, w=0.
    # t in [0, 1/3). s = 3*t.
    mask1 = t < (1.0/3.0)
    s1 = 3.0 * t[mask1]
    bary[mask1, 0] = 1.0 - s1
    bary[mask1, 1] = s1
    
    # Side 2: V1(0,1,0) to V2(0,0,1).
    # Param s in [0,1]. Bary: u=0, v=1-s, w=s.
    # t in [1/3, 2/3). s = 3*t - 1.
    mask2 = (t >= (1.0/3.0)) & (t < (2.0/3.0))
    s2 = 3.0 * t[mask2] - 1.0
    bary[mask2, 0] = 0.0
    bary[mask2, 1] = 1.0 - s2
    
    # Side 3: V2(0,0,1) to V0(1,0,0).
    # Param s in [0,1]. Bary: u=s, v=0, w=1-s.
    # t in [2/3, 1). s = 3*t - 2.
    mask3 = t >= (2.0/3.0)
    s3 = 3.0 * t[mask3] - 2.0
    bary[mask3, 0] = s3
    bary[mask3, 1] = 0.0
    
    return bary

def generate_perimeter_init():
    """
    Generate an initial configuration by placing points roughly evenly
    on the perimeter, with some random jitter.
    """
    rng = np.random.default_rng(42)
    # Start with evenly spaced points on perimeter
    t_vals = np.linspace(0, 1, N, endpoint=False)
    # Add random jitter
    jitter = rng.uniform(-0.05, 0.05, N)
    t_vals = (t_vals + jitter) % 1.0
    # Sort to maintain order? Not strictly necessary for area calc, but good for SA steps
    t_vals = np.sort(t_vals)
    
    bary = perimeter_to_bary(t_vals)
    return bary

def evaluate_t_config(t_vals):
    """
    Evaluate a configuration defined by perimeter parameters t_vals.
    """
    bary = perimeter_to_bary(t_vals)
    return min_area(bary)

def simulated_annealing_perimeter(t_init, time_limit):
    """
    Perform Simulated Annealing on the 1D perimeter parameter space.
    """
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    t_curr = t_init.copy()
    val_curr = evaluate_t_config(t_curr)
    
    t_best = t_curr.copy()
    val_best = val_curr
    
    T_start = 0.05
    T_end = 0.0001
    n_steps = 50000 # Will break based on time
    
    # Pre-calculate step sizes
    # Small moves
    step_small = 0.01
    # Large moves
    step_large = 0.05
    
    for i in range(n_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Cooling schedule
        progress = i / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Choose step size
        if rng.random() < 0.8:
            step_size = step_small
        else:
            step_size = step_large
            
        # Perturb one or two points
        n_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(N, n_perturb, replace=False)
        
        t_new = t_curr.copy()
        for idx in indices:
            t_new[idx] = (t_new[idx] + rng.normal(0, step_size)) % 1.0
            
        # Evaluate
        val_new = evaluate_t_config(t_new)
        
        # Acceptance
        if val_new > val_curr:
            t_curr = t_new
            val_curr = val_new
            if val_curr > val_best:
                t_best = t_curr.copy()
                val_best = val_curr
        else:
            delta_val = val_new - val_curr
            # Boltzmann factor
            if T > 0:
                prob = np.exp(delta_val / T)
                if rng.random() < prob:
                    t_curr = t_new
                    val_curr = val_new
                    
    return t_best, val_best

def refine_local_bary(bary_init, time_limit):
    """
    Final local refinement using SLSQP on the 22D barycentric space.
    """
    start_time = time.time()
    
    def objective(x):
        bary = x.reshape(N, 2)
        return -min_area(bary)
        
    def cons_u(x):
        return x[0::2] - 1e-10
        
    def cons_v(x):
        return x[1::2] - 1e-10
        
    def cons_sum(x):
        u = x[0::2]
        v = x[1::2]
        return (1 - 1e-10) - u - v
        
    constraints = [
        {'type': 'ineq', 'fun': cons_u},
        {'type': 'ineq', 'fun': cons_v},
        {'type': 'ineq', 'fun': cons_sum}
    ]
    
    bounds = [(0, 1)] * (N * 2)
    
    try:
        res = minimize(
            objective, 
            bary_init.flatten(), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        final_bary = clamp_bary(res.x.reshape(N, 2))
    except Exception:
        final_bary = bary_init
        
    return final_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Generate initial perimeter configuration
    t_init = generate_perimeter_init()
    
    # 2. Simulated Annealing on perimeter space
    # Allocate 60% of time to SA
    sa_time = 0.6 * (TIME_LIMIT - 5.0)
    t_best, val_best = simulated_annealing_perimeter(t_init, sa_time)
    
    # Convert best perimeter config to barycentric
    bary_sa = perimeter_to_bary(t_best)
    
    # 3. Local refinement in barycentric space
    # Allocate remaining time to SLSQP
    time_remaining = deadline - time.time()
    # Reserve 2s for final check
    refine_time = max(1.0, time_remaining - 2.0)
    
    bary_final = refine_local_bary(bary_sa, refine_time)
    
    # 4. Final validation and conversion to Cartesian
    cart = bary_to_cart(bary_final)
    
    return cart

# EVOLVE-BLOCK-END
