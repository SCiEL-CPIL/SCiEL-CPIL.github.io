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

def nlp_objective(x):
    """
    Objective function for NLP: Minimize -t (i.e., maximize t)
    x = [u0, v0, u1, v1, ..., u10, v10, t]
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    return -t

def nlp_constraints(x):
    """
    Constraints for NLP:
    1. Barycentric coordinates must be within the triangle (u>=0, v>=0, u+v<=1)
    2. Area of every triangle formed by points must be >= t
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    
    # Barycentric bounds are handled by SLSQP bounds, but let's add inequality constraints for safety
    # Actually, SLSQP handles simple bounds well. We need to add the area constraints.
    
    areas = calc_areas_bary(bary)
    
    # Constraint: areas - t >= 0 for all triangles
    # This is 165 constraints.
    return list(areas - t)

def solve_nlp(bary_init, t_init, time_limit=5.0):
    """
    Solve the NLP problem using SLSQP.
    """
    start_time = time.time()
    
    # Initial guess for t is the current min area
    x0 = np.concatenate([bary_init.flatten(), [t_init]])
    
    # Bounds for barycentric coords: [0, 1] for u and v
    # Note: u+v <= 1 is a linear constraint, but SLSQP handles simple bounds better.
    # We can add u+v <= 1 as a constraint or just rely on the fact that points are in the triangle.
    # For SLSQP, simple bounds are (0,1) for u and v. The constraint u+v<=1 must be added explicitly.
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # u_i
        bounds.append((0, 1)) # v_i
    bounds.append((0, 1)) # t (max area is 1)
    
    # Constraints:
    # 1. u_i + v_i <= 1 for all i
    # 2. Area_j >= t for all j
    
    constraints = []
    
    # Add u+v <= 1 constraints
    for i in range(N):
        def make_uv_constraint(i):
            def constraint(x):
                return 1.0 - x[2*i] - x[2*i+1]
            return {'type': 'ineq', 'fun': constraint}
        constraints.append(make_uv_constraint(i))
        
    # Add area >= t constraints
    # To avoid creating 165 lambda functions which is slow, we can vectorize if possible,
    # but SLSQP requires a list of dicts or a single dict with a vector fun.
    # Let's use a single constraint function that returns a vector.
    
    def area_constraints(x):
        bary = x[:N*2].reshape(N, 2)
        t = x[N*2]
        areas = calc_areas_bary(bary)
        return areas - t
        
    constraints.append({'type': 'ineq', 'fun': area_constraints})
    
    try:
        result = minimize(
            nlp_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < -t_init:
            # Extract best bary and t
            best_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
            best_t = result.x[N*2]
            
            # Verify feasibility and actual min area
            actual_min = min_area(best_bary)
            if actual_min > t_init:
                return best_bary, actual_min
        
    except Exception as e:
        pass
        
    return bary_init, t_init

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
        
    # 2. Solve NLP for each initial configuration
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        init_val = min_area(init_bary)
        update_best(init_bary)
        
        # Solve NLP
        t_init = init_val
        # Use a slightly smaller t_init to help the solver start feasible?
        # Actually, if we start with t = min_area, the constraints are active (0).
        # It's better to start with t slightly less than min_area to be strictly feasible.
        t_start = t_init * 0.99
        
        refined_bary, refined_val = solve_nlp(init_bary, t_start, time_limit=20.0)
        update_best(refined_bary)
        
    # 3. Refine the best found using multiple NLP restarts or higher precision
    if best_bary is not None:
        # Try a few more NLP runs from the best solution with different t starts
        for i in range(3):
            if time.time() > deadline - 5:
                break
            t_start = best_val * (0.95 + 0.05*i)
            refined_bary, refined_val = solve_nlp(best_bary, t_start, time_limit=10.0)
            update_best(refined_bary)
            
        # Final polish with SLSQP on the coordinates directly (maximize min area)
        # This is the "old" method but used as a polish for the NLP result
        if time.time() < deadline - 2:
            try:
                bounds = [(0, 1)] * (N * 2)
                # We need to add u+v<=1 constraints for the direct coordinate optimization too
                # Or just use the NLP formulation one last time with high precision
                result_final = minimize(
                    nlp_objective,
                    np.concatenate([best_bary.flatten(), [best_val * 0.99]]),
                    method='SLSQP',
                    bounds=bounds + [(0,1)],
                    constraints=[
                        {'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]} for i in range(N)
                    ] + [
                        {'type': 'ineq', 'fun': lambda x: calc_areas_bary(x[:N*2].reshape(N, 2)) - x[N*2]}
                    ],
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                if result_final.success:
                    final_bary = clamp_bary(result_final.x[:N*2].reshape(N, 2))
                    update_best(final_bary)
            except Exception:
                pass

    # Convert to cartesian
    if best_bary is None:
        best_bary = generate_random_init(42)
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
