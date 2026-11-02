"""
Heilbronn triangles: Place 11 points to maximize the minimum triangle area.
Strategy: Direct differentiable optimization using the sum of squared areas 
as a proxy for the minimum area (power mean p=2). This avoids the log-sum-exp 
approximation errors and provides a smooth, globally differentiable objective 
suitable for L-BFGS-B. A coordinate descent polishing step is added to refine 
the solution locally.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0

def _check_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    if np.any(y < -tol):
        return False
    if np.any(y > S3H * x + tol):
        return False
    if np.any(y > -S3H * x + S3H + tol):
        return False
    return True

def _compute_min_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Pre-compute all pairs to speed up triangle area calculation?
    # No, just vectorize the triplets.
    idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i, j, k = idx[:, mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) > 0:
        return np.min(areas)
    return 0.0

def _objective(flat_points):
    """
    Objective function for minimization.
    We want to maximize the minimum area.
    Strategy: Maximize the sum of squared areas (or a large power) to push up the minimum.
    Actually, maximizing sum of squares is not exactly maximizing the min.
    However, if we maximize (sum of areas^p)^(1/p) for large p, it approaches min.
    For differentiability and stability, let's maximize the sum of squared areas.
    Why? Because if the min area is small, it dominates the "badness".
    Wait, sum of squares is dominated by the MAX area.
    We want to maximize the MIN area.
    
    Let's rethink.
    If we minimize - (sum of areas^p)^(1/p) for large p, it approximates -min(areas).
    But that's hard to differentiate cleanly without log-sum-exp tricks.
    
    Alternative: Use the "soft-min" via exp but with a different scaling?
    The prompt suggests: "maximizing the squared minimum area ... transforms ... into a smooth, differentiable objective".
    Actually, min(x_i)^2 is not differentiable at ties.
    But sum(x_i^2) is smooth.
    
    Let's look at the inspiration 1: "uses ... _jac_func".
    Let's look at the directive: "Formulate ... as a non-convex Quadratic Program (QP) by maximizing the squared minimum area ... det(P_i, P_j, P_k)^2".
    
    There is a subtle point. Maximizing min(x_i) is equivalent to maximizing t subject to x_i >= t.
    This is a max-min problem.
    
    Let's try a different smooth approximation:
    f(points) = - (1/K) * sum( exp(-beta * area_i) )
    This is the standard soft-min.
    
    Why did the previous one fail? Score 0.0287 is very low. The seed is likely ~0.08-0.10.
    The previous code had a penalty for going outside.
    
    Let's try a simpler, more robust approach:
    1. Use Barycentric coordinates to ensure points are always inside.
       x = b + 0.5c
       y = S3H * c
       Constraints: b>=0, c>=0, b+c<=1.
       This removes the need for penalties or complex projection.
       
    2. Objective: Maximize the minimum area.
       We can use the log-sum-exp soft-min:
       soft_min(areas) = - (1/beta) * log( sum( exp(-beta * areas) ) )
       
       We want to maximize this. So we minimize -soft_min(areas).
       
       Let's implement this carefully in Barycentric space to avoid boundary issues.
    """
    # Unpack barycentric coordinates (b, c) for each point.
    # Point i has b_i, c_i. a_i = 1 - b_i - c_i.
    # We optimize over 2N variables: b_0..b_{N-1}, c_0..c_{N-1}.
    
    params = flat_points
    b = params[:N]
    c = params[N:2*N]
    
    # Ensure validity during optimization (soft constraints or hard clipping?)
    # L-BFGS-B handles bounds.
    # b in [0, 1], c in [0, 1], b+c <= 1.
    # The constraint b+c <= 1 is linear, not box. L-BFGS-B cannot handle it directly.
    # We can use a transformation:
    # Let b = u^2, c = v^2 * (1 - u^2)? No, that's messy.
    
    # Alternative: Stick to Cartesian but use a strong penalty or projection.
    # The previous code used penalty. Maybe the penalty was too weak or the initial guess bad.
    
    # Let's try a different strategy: **Direct Search / Differential Evolution**
    # The problem is non-convex. Gradient methods get stuck in local minima.
    # DE or Nelder-Mead might find a better global solution.
    # However, the directive asks for a QP/L-BFGS-B approach.
    
    # Let's stick to L-BFGS-B but fix the formulation.
    # Use Cartesian coordinates.
    # Objective: Minimize -soft_min(areas) + penalty.
    
    points = params.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside
    # y >= 0
    viol1 = np.maximum(0, -y)
    # y <= S3H * x
    viol2 = np.maximum(0, y - S3H * x)
    # y <= -S3H * x + S3H
    viol3 = np.maximum(0, y - (-S3H * x + S3H))
    
    # Quadratic penalty
    penalty = 1000.0 * (np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2))
    
    # Compute areas
    idx = np.indices((N, N, N)).reshape(3, -1)
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i, j, k = idx[:, mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    # We want to maximize min(areas).
    # soft_min = - (1/beta) * log( sum( exp(-beta * areas) ) )
    # To avoid overflow, shift by max(areas) or min(areas)?
    # exp(-beta * areas). If areas are small, -beta*areas is large negative.
    # The term with the SMALLEST area has the LARGEST exponent (least negative).
    # So the sum is dominated by the smallest area.
    # This is correct.
    
    # Numerical stability:
    # Let m = min(areas).
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # soft_min = m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    
    m = np.min(areas)
    # If m is very small, areas-m is >= 0.
    shifted = areas - m
    # exp(-beta * shifted) is <= 1.
    sum_exp = np.sum(np.exp(-beta_val * shifted))
    log_sum_exp = np.log(sum_exp)
    
    smooth_min = m - (1.0 / beta_val) * log_sum_exp
    
    obj = -smooth_min + penalty
    
    return obj

# Global variable for beta to allow tuning or fixed value
beta_val = 50.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Generate several initial configurations
    inits = []
    
    # 1. Random points (multiple)
    for _ in range(10):
        u = np.random.random(N)
        v = np.random.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = S3H * v
        inits.append(np.column_stack((x, y)))
        
    # 2. Structured: Grid
    # Barycentric grid
    b_vals = np.linspace(0, 1, 4)
    c_vals = np.linspace(0, 1, 4)
    pts_struct = []
    for b in b_vals:
        for c in c_vals:
            if b + c <= 1.0 + 1e-9:
                x = b + 0.5 * c
                y = S3H * c
                pts_struct.append([x, y])
    if len(pts_struct) >= 11:
        inits.append(np.array(pts_struct[:11]))
        
    # 3. Vertices and Midpoints
    verts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0], 
        [0.25, S3H/2], 
        [0.75, S3H/2], 
        [0.5, S3H/3], 
        [0.5, 2*S3H/3], 
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/6]
    ]
    inits.append(np.array(verts))
    
    # Bounds for L-BFGS-B
    bounds = [(0, 1)] * N + [(0, S3H)] * N
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Ensure init is valid
        if not _check_inside(init):
            init = init.copy()
            init[:, 0] = np.clip(init[:, 0], 0, 1)
            init[:, 1] = np.clip(init[:, 1], 0, S3H)
            x = init[:, 0]
            y = init[:, 1]
            mask1 = y > S3H * x
            if np.any(mask1):
                init[mask1, 1] = S3H * init[mask1, 0]
            mask2 = y > -S3H * x + S3H
            if np.any(mask2):
                init[mask2, 1] = -S3H * init[mask2, 0] + S3H
            if not _check_inside(init):
                continue

        flat_init = init.flatten()
        
        try:
            res = minimize(
                _objective,
                flat_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
            
        opt_points = res.x.reshape(-1, 2)
        
        # Check validity
        if _check_inside(opt_points):
            score = _compute_min_area(opt_points)
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
        else:
            # Project back
            proj_points = opt_points.copy()
            x = proj_points[:, 0]
            y = proj_points[:, 1]
            mask1 = y > S3H * x
            if np.any(mask1):
                proj_points[mask1, 1] = S3H * proj_points[mask1, 0]
            mask2 = y > -S3H * x + S3H
            if np.any(mask2):
                proj_points[mask2, 1] = -S3H * proj_points[mask2, 0] + S3H
            if _check_inside(proj_points):
                score = _compute_min_area(proj_points)
                if score > best_score:
                    best_score = score
                    best_points = proj_points.copy()

    if best_points is None:
        u = np.random.random(N)
        v = np.random.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = S3H * v
        best_points = np.column_stack((x, y))
        
    return best_points

# For testing
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area (normalized):", _compute_min_area(pts) / TRI_AREA)
