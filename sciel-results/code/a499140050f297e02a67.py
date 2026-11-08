"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can compute all areas for a fixed i by vectorizing j and k
    for i in range(n):
        pi = points[i]
        # Vectors from pi to all other points
        vecs = points - pi  # shape (n, 2)
        
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        # We need min of 0.5 * |C[j,k]| for j < k, j != i, k != i
        # Since C is antisymmetric, C[j,k] = -C[k,j]
        # We can take the upper triangle
        # Set diagonal and i-th row/col to inf to ignore them
        C_abs = np.abs(C)
        # Set i-th row and column to inf
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        # Set diagonal to inf (j==k area is 0, but we want distinct points)
        np.fill_diagonal(C_abs, float('inf'))
        
        # Find min in upper triangle
        # To be safe and fast, just find global min of C_abs (excluding i and diag)
        # Since it's symmetric, min in upper == min in lower
        current_min = np.min(C_abs)
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-9: # Early exit if degenerate
                return 0.0

    return (0.5 * min_area) / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _params_to_points(params):
    """Convert 2*N parameters to N points inside the triangle using barycentric coordinates."""
    # params shape (2*N,)
    # For each point, we use 2 params u, v such that u >= 0, v >= 0, u+v <= 1
    # To map from R^2 to the triangle domain, we can use sigmoid or tanh to bound u, v
    # Let's use u = sigmoid(p1), v = sigmoid(p2) * (1 - u)
    # This ensures u in (0,1) and v in (0, 1-u)
    points = []
    for i in range(N):
        u = 1.0 / (1.0 + np.exp(-params[2*i]))
        v = 1.0 / (1.0 + np.exp(-params[2*i+1])) * (1.0 - u)
        p = V0 * (1 - u - v) + V1 * u + V2 * v
        points.append(p)
    return np.array(points)


def _soft_min_area(params, beta=50.0):
    """
    Objective function for minimization: negative of the soft-minimum triangle area.
    Uses log-sum-exp trick to approximate min(a_i) as -1/beta * log(sum(exp(-beta * a_i))).
    This creates a smooth surrogate for the non-differentiable min function.
    """
    points = _params_to_points(params)
    n = len(points)
    
    # Collect all triangle areas
    # To save memory and time, we can compute them in chunks or just loop i
    # For N=11, total triangles is 165. We can store them all.
    areas = []
    
    for i in range(n):
        pi = points[i]
        vecs = points - pi
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        C_abs = np.abs(C)
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        
        # Get upper triangle values (j < k)
        # Since C is antisymmetric, C[j,k] = -C[k,j]. 
        # We want area = 0.5 * |C[j,k]| / AREA
        # We can just take the upper triangle of C_abs
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Exclude i
        mask[i, :] = False
        mask[:, i] = False
        
        vals = C_abs[mask]
        # Normalize to area
        # Area = 0.5 * val / AREA
        areas.extend((0.5 * vals) / AREA)
        
    areas = np.array(areas)
    
    # Soft min calculation
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    # Note: we are minimizing -soft_min, which is maximizing soft_min.
    # soft_min = -1/beta * logsumexp(-beta * areas)
    
    # If any area is 0, soft_min is 0.
    # We add a small epsilon to avoid log(0) issues if points coincide, 
    # though the mapping should prevent exact coincidence unless params are identical.
    areas_safe = np.maximum(areas, 1e-12)
    
    # LogSumExp: log(sum(exp(x)))
    # x = -beta * areas_safe
    x = -beta * areas_safe
    x_max = np.max(x)
    sum_exp = np.sum(np.exp(x - x_max))
    log_sum_exp = x_max + np.log(sum_exp)
    
    soft_min_val = -1.0 / beta * log_sum_exp
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return np.array(pts)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _points_to_params(points):
    """Inverse of _params_to_points. For initial params for optimization."""
    params = []
    for p in points:
        x, y = p
        v = 2 * y / np.sqrt(3)
        u = x - 0.5 * v
        # Inverse sigmoid
        # u = sigmoid(p1) => p1 = log(u/(1-u))
        # v = sigmoid(p2)*(1-u) => sigmoid(p2) = v/(1-u) => p2 = log( (v/(1-u)) / (1 - v/(1-u)) )
        # Handle boundaries
        eps = 1e-6
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        p1 = np.log(u / (1 - u))
        s2 = v / (1 - u)
        s2 = np.clip(s2, eps, 1-eps)
        p2 = np.log(s2 / (1 - s2))
        params.append([p1, p2])
    return np.array(params).flatten()


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy: Multi-start L-BFGS-B with soft-min surrogate
    # 1. Start with structured initial
    # 2. Start with several random initializations
    
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(15):
        initial_configs.append(_generate_random_points(rng))
    
    # Run L-BFGS-B on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Convert to params
        init_params = _points_to_params(init_pts)
        
        # L-BFGS-B settings
        # maxiter is important to prevent hanging
        # We want to give it enough iterations to converge
        # Since we have multiple starts, we can limit iterations per start
        # L-BFGS-B is generally faster per iteration than NM for smooth problems
        max_iter_per_start = 200 if idx == 0 else 100
        
        try:
            # Use a moderate beta for the soft-min to balance smoothness and accuracy
            # Lower beta = smoother, higher beta = closer to true min
            # Start with beta=50, could increase if we have time for polish
            res = minimize(
                _soft_min_area, 
                init_params, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter_per_start, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            # Evaluate the true objective on the result
            best_params = res.x
            best_points_candidate = _params_to_points(best_params)
            current_min = _min_triangle_area(best_points_candidate)
            
            if current_min > best_min:
                best_min = current_min
                best_points = best_points_candidate.copy()
                
                # If we found a good solution, we can try to polish it with higher precision
                # or just move on to next start
                # Given time constraints, we'll just move on unless we have lots of time left
                if time.time() < deadline - 20 and best_min > 0.01:
                    # Optional: quick local polish with higher beta or NM on the result
                    # But NM might be slower. Let's stick to L-BFGS-B for now.
                    pass
        except Exception as e:
            continue

    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
# EVOLVE-BLOCK-END
