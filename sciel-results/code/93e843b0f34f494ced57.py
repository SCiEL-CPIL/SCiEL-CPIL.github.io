import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2.0
TRI_AREA = H * 1.0 / 2.0 # Area of equilateral triangle with side 1

# Precompute all triples for vectorization
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
_i_idx = _idx_arr[:, 0]
_j_idx = _idx_arr[:, 1]
_k_idx = _idx_arr[:, 2]

def _project_to_triangle(points):
    """
    Project points (n, 2) into the triangle with vertices (0,0), (1,0), (0.5, H).
    Uses barycentric coordinates for robust projection.
    """
    # Barycentric coordinates relative to V0=(0,0), V1=(1,0), V2=(0.5, H)
    # P = b0*V0 + b1*V1 + b2*V2
    # x = b1 + 0.5*b2
    # y = H*b2
    # => b2 = y/H
    # => b1 = x - 0.5*b2
    # => b0 = 1 - b1 - b2
    
    x = points[:, 0]
    y = points[:, 1]
    
    b2 = y / H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    
    # Clip to valid simplex [0, 1]
    b0 = np.clip(b0, 0.0, 1.0)
    b1 = np.clip(b1, 0.0, 1.0)
    b2 = np.clip(b2, 0.0, 1.0)
    
    # Normalize to ensure sum is 1 (if clipping changed the sum)
    sum_b = b0 + b1 + b2
    # Avoid division by zero
    sum_b = np.where(sum_b == 0, 1.0, sum_b)
    b0 /= sum_b
    b1 /= sum_b
    b2 /= sum_b
    
    # Convert back to Cartesian
    px = b1 * 1.0 + b2 * 0.5
    py = b2 * H
    
    return np.column_stack((px, py))

def _compute_areas(points):
    """
    Compute areas for all C(N,3) triangles.
    points: (N, 2)
    Returns: array of shape (165,)
    """
    # Vectorized cross product calculation
    # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    xi = points[_i_idx, 0]
    yi = points[_i_idx, 1]
    xj = points[_j_idx, 0]
    yj = points[_j_idx, 1]
    xk = points[_k_idx, 0]
    yk = points[_k_idx, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area_exact(points):
    """Calculate the true minimum area."""
    return np.min(_compute_areas(points))

def _soft_min_objective(params, temperature=0.001):
    """
    Objective function for local optimization.
    Minimizes the negative soft-min of triangle areas.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points)
    
    areas = _compute_areas(points)
    # Avoid log(0) or exp overflow
    areas = np.maximum(areas, 1e-12)
    
    # Soft min: -T * log(sum(exp(-A/T)))
    # For stability:
    min_a = np.min(areas)
    # If min_a is very small, the soft min is dominated by it.
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -temperature * log_sum
    
    return -soft_min_val

def _soft_min_gradient(params, temperature=0.001):
    """
    Gradient of the soft-min objective.
    d(-soft_min)/dP = sum_k w_k * d(A_k)/dP
    where w_k = exp(-A_k/T) / sum(exp(-A_j/T))
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points)
    
    areas = _compute_areas(points)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp # These are d(soft_min)/d(A_k)
    
    # We want gradient of -soft_min, so -weights * d(A_k)/dP
    
    # Gradient of Area w.r.t points
    # A_k = 0.5 * |det_k|
    # Let s_k = det_k = x_i(y_j - y_k) + ...
    # d(s_k)/dx_i = y_j - y_k
    # d(s_k)/dy_i = x_k - x_j
    # d(s_k)/dx_j = y_k - y_i
    # d(s_k)/dy_j = x_i - x_k
    # d(s_k)/dx_k = y_i - y_j
    # d(s_k)/dy_k = x_j - x_i
    
    # Note: A = 0.5 * |s|. dA/dP = 0.5 * sign(s) * ds/dP.
    # If s is close to 0, sign is undefined, but weight w_k is also small if T is small?
    # Actually, if s is small, A is small, so exp(-A/T) is large, so w_k is large.
    # This creates a "spike" in gradient. This is expected for max-min problems.
    
    # To handle the absolute value non-differentiability at 0 smoothly,
    # we can use the fact that the soft-min smooths the min, but the area calculation
    # itself has a kink at 0.
    # A common trick is to use A = 0.5 * sqrt(s^2 + eps) or just accept the kink
    # if the optimizer is robust. L-BFGS-B can handle kinks if we provide a
    # consistent subgradient. The sign(s) is a valid subgradient.
    
    s_vals = 2.0 * areas # Since areas = 0.5 * |s|, |s| = 2*areas. But we need sign.
    # Let's recompute signed determinants
    xi = points[_i_idx, 0]
    yi = points[_i_idx, 1]
    xj = points[_j_idx, 0]
    yj = points[_j_idx, 1]
    xk = points[_k_idx, 0]
    yk = points[_k_idx, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    
    # Sign of determinant
    # If det is 0, gradient is 0 (or undefined). We use 0 to avoid NaNs.
    signs = np.sign(det)
    # For very small det, the area is dominated by numerical noise.
    # We can zero out gradients for triangles with area < threshold to reduce noise
    # or just use the sign.
    
    grad_points = np.zeros_like(points)
    
    # d(A_k)/dx_i = 0.5 * sign_k * (y_j - y_k)
    # d(A_k)/dy_i = 0.5 * sign_k * (x_k - x_j)
    # etc.
    
    # Vectorized accumulation
    # For each triangle k with weight w_k:
    # grad_x_i += w_k * 0.5 * sign_k * (y_j - y_k)
    # grad_y_i += w_k * 0.5 * sign_k * (x_k - x_j)
    
    w_half = 0.5 * weights
    
    # Accumulate for i
    grad_points[_i_idx, 0] += w_half * signs * (yj - yk)
    grad_points[_i_idx, 1] += w_half * signs * (xk - xj)
    
    # Accumulate for j
    grad_points[_j_idx, 0] += w_half * signs * (yk - yi)
    grad_points[_j_idx, 1] += w_half * signs * (xi - xk)
    
    # Accumulate for k
    grad_points[_k_idx, 0] += w_half * signs * (yi - yj)
    grad_points[_k_idx, 1] += w_half * signs * (xj - xi)
    
    # The objective is -soft_min.
    # d(-soft_min)/dP = - sum w_k d(A_k)/dP
    # So we negate the gradient of soft_min (which is sum w_k dA/dP)
    # Wait, my weights w_k were defined as d(soft_min)/d(A_k).
    # So grad(soft_min) = sum w_k dA/dP.
    # grad(-soft_min) = -grad(soft_min).
    
    return -grad_points.flatten()

def _local_polish(points, max_iter=500, time_limit=10.0):
    """
    Polish a configuration using L-BFGS-B.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    # Bounds: x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    # Use a small temperature for the soft-min to approximate the true min closely
    # during polishing.
    temp = 0.0005
    
    def fun(x):
        if time.time() - start_time > time_limit:
            return _soft_min_objective(x, temp) + 1e10 # Penalize timeout
        return _soft_min_objective(x, temp)
        
    def jac(x):
        if time.time() - start_time > time_limit:
            return np.zeros_like(x)
        return _soft_min_gradient(x, temp)
        
    try:
        result = minimize(
            fun,
            x0,
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        polished_points = result.x.reshape(N, 2)
        polished_points = _project_to_triangle(polished_points)
        return polished_points
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 150s kill
    
    np.random.seed(42)
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust to local minima.
    # We use a subset of the time for DE.
    de_time_budget = 60.0
    
    # Bounds for DE
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    # Objective for DE: We want to maximize min area.
    # DE minimizes. So we minimize -min_area.
    # However, -min_area is non-differentiable. DE doesn't need gradients.
    # But to avoid getting stuck in "flat" regions of the min function,
    # we can use the soft-min objective with a moderate temperature for DE,
    # or just the exact min area. Exact min area is better for the final goal.
    # Let's use exact min area for DE to directly target the goal.
    
    def de_objective(params):
        if time.time() - start_time > start_time + de_time_budget:
            return 1e10
        points = params.reshape(N, 2)
        points = _project_to_triangle(points)
        min_a = _min_area_exact(points)
        return -min_a
    
    # Run DE
    # popsize=10, maxiter=100. 
    # DE is stochastic.
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-6,
            polish=False, # We will do our own polish
            init='sobol' # Better space-filling than 'latinhypercube' or 'random'
        )
        best_de_points = de_result.x.reshape(N, 2)
        best_de_points = _project_to_triangle(best_de_points)
        
        # Collect candidates from DE history? 
        # DE only returns the best. 
        # To get multiple candidates, we can run DE multiple times or 
        # extract from the population. 
        # Simpler: Run DE once to get a good start, then do random restarts 
        # around it and from scratch.
        
    except Exception:
        best_de_points = np.random.rand(N, 2)
        best_de_points[:, 0] *= 1.0
        best_de_points[:, 1] *= H
        best_de_points = _project_to_triangle(best_de_points)
        
    # Phase 2: Local Polish of the DE result
    # And several random restarts.
    
    candidates = []
    
    # 1. Polish the DE result
    polished_de = _local_polish(best_de_points, max_iter=200, time_limit=15.0)
    candidates.append(polished_de)
    
    # 2. Random restarts
    # If we have time left, do random restarts with polish.
    while time.time() < deadline - 20.0:
        # Generate random points
        rand_points = np.random.rand(N, 2)
        rand_points[:, 0] *= 1.0
        rand_points[:, 1] *= H
        rand_points = _project_to_triangle(rand_points)
        
        # Polish
        polished_rand = _local_polish(rand_points, max_iter=100, time_limit=5.0)
        candidates.append(polished_rand)
        
    # 3. Also try polishing the DE result again with a tighter temperature
    # or more iterations if time permits.
    if time.time() < deadline - 10.0:
        polished_de2 = _local_polish(best_de_points, max_iter=500, time_limit=10.0)
        candidates.append(polished_de2)
    
    # Select the best candidate based on EXACT minimum area
    best_points = None
    best_score = 0.0
    
    for pts in candidates:
        if pts is None:
            continue
        score = _min_area_exact(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Fallback
    if best_points is None:
        best_points = _project_to_triangle(np.random.rand(N, 2) * np.array([1.0, H]))
        
    return best_points

# EVOLVE-BLOCK-END
