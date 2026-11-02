"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

# Constants
TIME_LIMIT = 115.0  # Safety margin for 120s limit
START_TIME = time.time()

def check_time():
    return time.time() - START_TIME < TIME_LIMIT

def bary_to_cartesian(bary_coords):
    """
    Convert barycentric coordinates (n, 3) to Cartesian coordinates (n, 2).
    Reference triangle vertices:
    A = (0, 0)
    B = (1, 0)
    C = (0.5, sqrt(3)/2)
    P = u*A + v*B + w*C
    """
    u, v, w = bary_coords[:, 0], bary_coords[:, 1], bary_coords[:, 2]
    x = 0 * u + 1 * v + 0.5 * w
    y = 0 * u + 0 * v + (np.sqrt(3) / 2) * w
    return np.column_stack((x, y))

def compute_min_area_softmin(bary_coords, alpha):
    """
    Compute the soft-minimum area of all triangles formed by the points.
    Returns negative soft-min area for minimization.
    """
    n = len(bary_coords)
    # Compute areas for all triplets
    # Area is 0.5 * |det|, but since we want to maximize min area, 
    # and scaling is constant, we can just use |det|.
    # Actually, the task says "maximize the smallest area".
    # Let's compute all pairwise determinants.
    
    # Efficient calculation of all triangle areas
    # We can use the fact that area(P_i, P_j, P_k) = 0.5 * | (P_j - P_i) x (P_k - P_i) |
    # In barycentric coordinates relative to a unit area triangle, 
    # the area ratio is exactly |det([u_i v_i w_i; u_j v_j w_j; u_k v_k w_k])|
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = np.linalg.det(bary_coords[[i, j, k]])
                areas.append(abs(det))
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * areas)))
    # To avoid overflow, subtract max
    # We want to minimize -softmin, which is equivalent to maximizing softmin.
    # Let's return the negative of the soft-min value for the optimizer.
    # softmin = -1/alpha * log(sum(exp(-alpha * areas)))
    # We want to maximize this, so we minimize -softmin.
    
    # Optimization: If alpha is large, exp(-alpha * area) can underflow.
    # We can use the log-sum-exp trick.
    # However, for small alpha, it's fine.
    
    # Actually, simpler:
    # max_area = np.max(areas)
    # shifted = areas - max_area
    # log_sum = max_area + np.log(np.sum(np.exp(-alpha * shifted)))
    # softmin = -1/alpha * log_sum
    
    # But we want to MINIMIZE the negative softmin.
    # Objective = -softmin = 1/alpha * log(sum(exp(-alpha * areas)))
    
    # Let's just compute it directly with care.
    # If alpha is small, exp(-alpha * area) is close to 1.
    # If alpha is large, only the smallest area contributes.
    
    # Using log-sum-exp for stability:
    # log(sum(exp(-alpha * a_i))) = -alpha * max(a_i) + log(sum(exp(-alpha * (a_i - max(a_i)))))
    # This is wrong. exp(-alpha * a_i) is largest when a_i is smallest.
    # Let a_min = min(areas).
    # exp(-alpha * a_i) = exp(-alpha * a_min) * exp(-alpha * (a_i - a_min))
    # sum = exp(-alpha * a_min) * sum(exp(-alpha * (a_i - a_min)))
    # log(sum) = -alpha * a_min + log(sum(exp(-alpha * (a_i - a_min))))
    # softmin = -1/alpha * (-alpha * a_min + log_sum_shifted) = a_min - 1/alpha * log_sum_shifted
    
    a_min = np.min(areas)
    shifted = areas - a_min
    # exp(-alpha * shifted) will be <= 1
    log_sum_shifted = np.log(np.sum(np.exp(-alpha * shifted)))
    softmin_val = a_min - (1.0 / alpha) * log_sum_shifted
    
    # We want to maximize softmin_val, so we return -softmin_val
    return -softmin_val

def gradient_softmin(bary_coords, alpha):
    """
    Analytic gradient of the soft-min objective with respect to barycentric coordinates.
    This is complex. We might rely on numerical gradient or a simpler approach.
    Given the time constraint and complexity, let's try a simpler multi-start local search
    with Nelder-Mead or BFGS on the Cartesian coordinates with a penalty for being outside,
    OR just use the barycentric soft-min with numerical gradients if analytic is too hard.
    
    Actually, let's stick to the plan: Barycentric + Soft-Min.
    We can use `scipy.optimize.minimize` with `method='L-BFGS-B'` and `jac='2-point'` 
    if analytic gradient is not easily derived. 
    However, for 11 points (33 variables), numerical gradient is feasible.
    """
    pass

def optimize_configuration(initial_bary, alpha_schedule):
    """
    Run the multi-stage optimization starting from initial_bary.
    """
    current_bary = initial_bary.copy()
    
    for alpha in alpha_schedule:
        if not check_time():
            break
            
        def objective(x):
            # x is flattened barycentric coords
            bary = x.reshape(-1, 3)
            # Normalize barycentric coords to sum to 1? 
            # L-BFGS-B with bounds [0, 1] on u, v, and w = 1 - u - v is better.
            # Let's represent state as (u, v) for each point, w = 1 - u - v.
            # This reduces dimensionality to 2*N.
            pass
            
        # Let's refine the representation: 
        # Use (u, v) for each point. w is implicit.
        # Bounds: u >= 0, v >= 0, u + v <= 1.
        # This is a simplex constraint.
        
        # Let's switch to optimizing (u, v) directly.
        # current_bary is (N, 3). We extract u, v.
        uv = current_bary[:, :2].flatten()
        
        def obj_and_grad(uv_flat):
            uv = uv_flat.reshape(-1, 2)
            w = 1.0 - uv[:, 0] - uv[:, 1]
            bary = np.column_stack((uv[:, 0], uv[:, 1], w))
            
            # Check validity (should be enforced by bounds, but softmin needs valid points)
            # If any w < 0, it's outside. 
            # We will use bounds in L-BFGS-B to keep u, v in [0, 1] and u+v <= 1?
            # L-BFGS-B doesn't handle linear constraints like u+v<=1 directly.
            # It handles box constraints.
            # So we need a transformation or a penalty.
            
            # Alternative: Use a change of variables that maps a box to a simplex?
            # Or just use a penalty in the objective.
            # Given the "soft-min" insight, maybe the landscape is smooth enough.
            
            # Let's try a different approach: 
            # Optimize in Cartesian coordinates with a penalty for being outside the triangle.
            # This is simpler to bound (box constraints on x, y).
            # Triangle: y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*(x-1).
            # Wait, the triangle is (0,0), (1,0), (0.5, sqrt(3)/2).
            # Lines:
            # 1. y = 0
            # 2. y = sqrt(3) x
            # 3. y = -sqrt(3) (x - 1) = sqrt(3) (1 - x)
            
            # Penalty if outside.
            
            pass

    # This path is getting complicated. Let's go back to the successful insight:
    # "Barycentric representation naturally enforces triangle constraints"
    # How? If we optimize u, v, w with u+v+w=1, we need to handle the equality constraint.
    # Or, we optimize u, v and set w=1-u-v, and use bounds u>=0, v>=0, u+v<=1.
    # `scipy.optimize.minimize` with `SLSQP` handles linear constraints.
    # `L-BFGS-B` does not.
    # Let's use `SLSQP` or `COBYLA`. `COBYLA` is robust for constraints.
    
    # Let's restart the optimization function design.
    pass

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initialize with a few random valid configurations
    best_bary = None
    best_score = -np.inf
    
    # Reference triangle area is 1.
    # We want to maximize min area.
    
    # Alpha schedule: start smooth, end sharp
    alpha_schedule = [10, 50, 100, 200, 500, 1000]
    
    # Number of restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if not check_time():
            break
            
        # Random initialization in barycentric coords
        # u, v, w > 0, sum = 1
        # Generate random points in the simplex
        u = np.random.rand(N)
        v = np.random.rand(N)
        # Normalize to ensure sum u+v+w=1? 
        # No, we need each point to be in the simplex.
        # For each point, generate 3 random numbers, normalize.
        init_bary = np.random.rand(N, 3)
        init_bary /= init_bary.sum(axis=1, keepdims=True)
        
        # Optimization function
        def objective_and_grad(x_flat):
            # x_flat is 2*N variables (u, v)
            uv = x_flat.reshape(N, 2)
            w = 1.0 - uv[:, 0] - uv[:, 1]
            bary = np.column_stack((uv[:, 0], uv[:, 1], w))
            
            # Penalty for being outside (w < 0 or u < 0 or v < 0)
            # Since we use SLSQP with constraints, this shouldn't happen, 
            # but for safety in the objective:
            min_coord = np.min(bary)
            if min_coord < -1e-6:
                return 1e6 - min_coord, None # Large penalty
            
            # Compute soft-min area
            # For gradient, we need analytic or numerical.
            # Let's use numerical gradient via finite differences in SLSQP?
            # SLSQP can use numerical gradients if jac is not provided.
            
            val = compute_min_area_softmin(bary, current_alpha)
            return val, None
        
        # We need to wrap the objective to use the current alpha from the loop
        # This is tricky with closures. Let's define a class or use a mutable container.
        alpha_holder = [alpha_schedule[0]]
        
        def obj_wrapper(x_flat):
            uv = x_flat.reshape(N, 2)
            w = 1.0 - uv[:, 0] - uv[:, 1]
            bary = np.column_stack((uv[:, 0], uv[:, 1], w))
            
            # Simple penalty for invalid barycentric coords (should be handled by constraints)
            if np.any(bary < -1e-9):
                return 1e6
            # Clip to 0 for stability in log/exp
            bary_clipped = np.clip(bary, 1e-9, None)
            # Renormalize? No, if they sum to 1, clipping breaks it.
            # Better to rely on constraints.
            
            val = compute_min_area_softmin(bary, alpha_holder[0])
            return val
        
        # Constraints for SLSQP:
        # w = 1 - u - v >= 0  =>  u + v <= 1
        # u >= 0, v >= 0
        
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1.0 - x[0::2].sum() - x[1::2].sum()}, # This is wrong, sum is over points? No.
            # Constraint is per point: u_i + v_i <= 1
            # SLSQP constraints are for the whole vector.
            # We need N constraints: u_i + v_i - 1 <= 0
            # fun(x) returns array of constraint values? 
            # SLSQP 'ineq' constraint: fun(x) >= 0.
            # So we want 1 - u_i - v_i >= 0.
            # We can return a vector of these values.
            {'type': 'ineq', 'fun': lambda x: 1.0 - x[0::2] - x[1::2]}
        ]
        
        # Bounds: u_i >= 0, v_i >= 0
        bounds = [(0, 1)] * (2 * N)
        
        # Initial guess for SLSQP: u, v from init_bary
        init_uv = init_bary[:, :2].flatten()
        
        for alpha in alpha_schedule:
            if not check_time():
                break
            alpha_holder[0] = alpha
            
            try:
                res = minimize(
                    obj_wrapper, 
                    init_uv, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.success:
                    init_uv = res.x
                else:
                    # If it fails, keep previous best or break
                    pass
                    
            except Exception as e:
                print(f"Optimization error: {e}")
                break
        
        # Evaluate final result
        final_uv = init_uv.reshape(N, 2)
        final_w = 1.0 - final_uv[:, 0] - final_uv[:, 1]
        final_bary = np.column_stack((final_uv[:, 0], final_uv[:, 1], final_w))
        
        # Check validity
        if np.any(final_bary < -1e-6):
            continue
            
        # Compute true min area (alpha -> inf)
        # Use a very large alpha for evaluation
        final_score = -compute_min_area_softmin(final_bary, 10000)
        
        if final_score > best_score:
            best_score = final_score
            best_bary = final_bary.copy()
            
    # If no valid solution found, return random
    if best_bary is None:
        # Fallback: random points
        pts = np.random.rand(N, 3)
        pts /= pts.sum(axis=1, keepdims=True)
        best_bary = pts
        
    # Convert to Cartesian
    cart_points = bary_to_cartesian(best_bary)
    
    return cart_points

# EVOLVE-BLOCK-END
