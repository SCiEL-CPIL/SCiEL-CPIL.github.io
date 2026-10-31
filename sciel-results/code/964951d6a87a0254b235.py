import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution, minimize

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_areas_bary(bary):
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

def objective_constrained(x):
    """
    x = [t, u1, v1, ..., u11, v11]
    Maximize t (minimize -t)
    Subject to: Area(i,j,k) >= t
    """
    t = x[0]
    bary = x[1:].reshape(N, 2)
    
    # Check if bary is valid
    if np.any(bary < -1e-6) or np.any(bary > 1 + 1e-6):
        return 1e10
    if np.any(1 - bary[:, 0] - bary[:, 1] < -1e-6):
        return 1e10
        
    areas = calc_areas_bary(bary)
    
    # Penalty if any area is less than t
    # We want to minimize -t, but we need to satisfy areas >= t.
    # If areas < t, we add a penalty.
    violation = np.sum(np.maximum(0, t - areas)) ** 2
    return -t + violation

def constraints_constrained(x):
    """
    Define constraints for SLSQP.
    g(x) >= 0
    """
    t = x[0]
    bary = x[1:].reshape(N, 2)
    
    # Clamp for calculation stability, but SLSQP should handle bounds
    bary_clamped = clamp_bary(bary)
    areas = calc_areas_bary(bary_clamped)
    
    # Constraints: Area - t >= 0
    return areas - t

def bounds_constrained():
    """
    Bounds for SLSQP.
    t is unbounded below in theory, but area is small. Let's bound t to [0, 0.1].
    Bary coords: u, v in [0, 1], u+v <= 1.
    SLSQP handles linear constraints better, but we can use bounds for u,v and add u+v<=1 as a constraint?
    Actually, it's easier to just bound u, v in [0, 1] and rely on the fact that the objective/penalty will push valid solutions.
    However, SLSQP is strict. Let's use bounds [0, 1] for u and v.
    The constraint u+v<=1 is not automatically enforced by bounds [0,1]x[0,1].
    We must add it as a constraint or use a transformation.
    Let's add u+v-1 <= 0 as a constraint.
    """
    # t bound
    t_bounds = (0.0, 0.1)
    # bary bounds
    bary_bounds = [(0.0, 1.0)] * (2 * N)
    return [t_bounds] + bary_bounds

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    # 1. Global Search: Differential Evolution
    # We search in Cartesian or Barycentric? Barycentric is constrained.
    # DE doesn't handle inequality constraints like u+v<=1 well without penalty.
    # Let's use a penalty function for DE in Barycentric space.
    
    def de_objective(x):
        bary = x.reshape(N, 2)
        # Penalty for invalid barycentric coords
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1 - u - v
        if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
            return 1e10
        areas = calc_areas_bary(bary)
        return -np.min(areas)
        
    # Bounds for DE: u, v in [0, 1]
    de_bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        # Run DE with limited iterations to stay within time
        # popsize=15, maxiter=50 is a good start
        de_result = differential_evolution(
            de_objective,
            bounds=de_bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            polish=False, # We will polish separately
            init='sobol'
        )
        
        if time.time() > deadline - 20:
            # Time is up, use DE result directly
            best_bary = clamp_bary(de_result.x.reshape(N, 2))
            best_val = -de_objective(de_result.x)
            cart = bary_to_cart(best_bary)
            return cart
            
        best_bary = clamp_bary(de_result.x.reshape(N, 2))
        best_val = -de_objective(de_result.x)
        
    except Exception:
        # Fallback to random init
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        best_val = -np.min(calc_areas_bary(best_bary))

    # 2. Local Polish: SLSQP on Constrained Problem
    # Initial guess: [t, bary]
    # t should be slightly less than the current min area to be feasible
    min_area = np.min(calc_areas_bary(best_bary))
    x0 = np.concatenate(([min_area * 0.9], best_bary.flatten()))
    
    # Define constraints for SLSQP
    # 1. Area constraints: areas - t >= 0
    # 2. Barycentric sum constraint: u_i + v_i - 1 <= 0  =>  1 - u_i - v_i >= 0
    
    def slsqp_constraints(x):
        t = x[0]
        bary = x[1:].reshape(N, 2)
        areas = calc_areas_bary(bary)
        
        # Area constraints
        c_areas = areas - t
        
        # Barycentric sum constraints (w >= 0)
        w_vals = 1.0 - bary[:, 0] - bary[:, 1]
        c_bary = w_vals
        
        return np.concatenate([c_areas, c_bary])
        
    # SLSQP expects g(x) >= 0
    cons = [{'type': 'ineq', 'fun': slsqp_constraints}]
    
    bounds = bounds_constrained()
    
    try:
        if time.time() < deadline - 5:
            res = minimize(
                objective_constrained,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < objective_constrained(x0):
                x_opt = res.x
                t_opt = x_opt[0]
                bary_opt = x_opt[1:].reshape(N, 2)
                
                # Verify
                bary_opt = clamp_bary(bary_opt)
                min_area_opt = np.min(calc_areas_bary(bary_opt))
                
                if min_area_opt > best_val:
                    best_val = min_area_opt
                    best_bary = bary_opt
                    
    except Exception:
        pass

    # 3. Final Polish: Nelder-Mead on the best candidate found
    if time.time() < deadline - 2:
        def nm_objective(bary_flat):
            bary = clamp_bary(bary_flat.reshape(N, 2))
            areas = calc_areas_bary(bary)
            return -np.min(areas)
            
        res_nm = minimize(
            nm_objective,
            best_bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if res_nm.fun < -best_val:
            best_val = -res_nm.fun
            best_bary = clamp_bary(res_nm.x.reshape(N, 2))

    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
