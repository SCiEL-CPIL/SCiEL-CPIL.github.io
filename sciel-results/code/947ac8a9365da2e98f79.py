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

# Precompute all unique triplets of indices (i, j, k) with i < j < k
# There are C(11, 3) = 165 triplets
TRIPLETS = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIPLETS.append((i, j, k))
TRIPLETS = np.array(TRIPLETS) # shape (165, 3)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    for i in range(n):
        pi = points[i]
        vecs = points - pi  # shape (n, 2)
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        C_abs = np.abs(C)
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        
        current_min = np.min(C_abs)
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-9:
                return 0.0

    return (0.5 * min_area) / AREA


def _params_to_points(params):
    """Convert 2*N parameters to N points inside the triangle using barycentric coordinates."""
    points = []
    for i in range(N):
        u = 1.0 / (1.0 + np.exp(-params[2*i]))
        v = 1.0 / (1.0 + np.exp(-params[2*i+1])) * (1.0 - u)
        p = V0 * (1 - u - v) + V1 * u + V2 * v
        points.append(p)
    return np.array(points)


def _points_to_params(points):
    """Inverse of _params_to_points."""
    params = []
    for p in points:
        x, y = p
        v = 2 * y / np.sqrt(3)
        u = x - 0.5 * v
        eps = 1e-6
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        p1 = np.log(u / (1 - u))
        s2 = v / (1 - u)
        s2 = np.clip(s2, eps, 1-eps)
        p2 = np.log(s2 / (1 - s2))
        params.append([p1, p2])
    return np.array(params).flatten()


def _soft_min_area(params, beta=50.0):
    """
    Compute a smooth surrogate for the negative of the minimum triangle area.
    Uses Log-Sum-Exp to approximate the minimum.
    We want to maximize min(area). So we minimize -min(area).
    min(area) ~ -1/beta * log(sum(exp(-beta * area)))
    So objective is: 1/beta * log(sum(exp(-beta * area)))
    Note: area here is unnormalized cross product magnitude (2*Area).
    We will normalize by AREA at the end if needed, but for optimization scale doesn't matter much.
    """
    points = _params_to_points(params)
    
    # Calculate areas for all triplets
    # points shape (N, 2)
    # For each triplet (i, j, k), area = 0.5 * | (pj-pi) x (pk-pi) |
    # Let's compute cross products for all triplets at once using indexing.
    
    # Extract points for i, j, k
    pi = points[TRIPLETS[:, 0]]
    pj = points[TRIPLETS[:, 1]]
    pk = points[TRIPLETS[:, 2]]
    
    # Vectors
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (scalar in 2D)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Area = 0.5 * |cross|
    areas = 0.5 * np.abs(cross)
    
    # Normalize by triangle area for consistency with _min_triangle_area
    areas = areas / AREA
    
    # Soft min
    # To avoid overflow in exp(-beta * area), we can subtract max area?
    # Wait, we want min.
    # min(x) = -max(-x)
    # -max(-x) = -1/beta * log(sum(exp(-beta * -x))) ? No.
    # Softmin(x, beta) = -1/beta * log(sum(exp(-beta * x)))
    # We want to MINIMIZE this softmin.
    
    # To ensure numerical stability:
    # Let M = max(areas).
    # sum(exp(-beta * areas)) = exp(-beta * M) * sum(exp(-beta * (areas - M)))
    # log(sum(...)) = -beta * M + log(sum(exp(-beta * (areas - M))))
    # Softmin = -1/beta * (-beta * M + log(...)) = M - 1/beta * log(...)
    
    # However, areas are positive. If beta is large, exp(-beta*area) is small for large areas.
    # The term with the smallest area dominates the sum.
    # Let's just use np.logsumexp for stability.
    
    # We want to minimize the softmin.
    # If we use logsumexp(-beta * areas), it computes log(sum(exp(-beta * areas))) stably.
    # softmin = -1/beta * logsumexp(-beta * areas)
    
    try:
        log_sum_exp_val = np.logaddexp.reduce(-beta * areas)
        soft_min = -1.0 / beta * log_sum_exp_val
    except:
        # Fallback to hard min if something goes wrong
        soft_min = np.min(areas)
        
    return soft_min


def _objective_smooth(params, beta=50.0):
    """Objective for L-BFGS-B: smooth surrogate of min area."""
    return _soft_min_area(params, beta)


def _objective_hard(params):
    """Objective for Nelder-Mead: true min area."""
    points = _params_to_points(params)
    return -_min_triangle_area(points)


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


def _generate_structured_initial():
    """Generate a structured initial configuration."""
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


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Initial configurations
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(8):
        initial_configs.append(_generate_random_points(rng))
    
    # Strategy:
    # 1. Use L-BFGS-B with a smooth surrogate (Log-Sum-Exp) to quickly find a good basin.
    # 2. Polish with Nelder-Mead on the true (non-smooth) objective.
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        init_params = _points_to_params(init_pts)
        
        # Phase 1: Smooth optimization with L-BFGS-B
        # We need to define a function that takes params and returns the soft min area.
        # L-BFGS-B minimizes. So we minimize the soft min area.
        
        # We can use a moderate beta. Too high -> non-smooth. Too low -> poor approximation.
        # Let's start with beta=20 and maybe increase it? Or just keep it constant.
        # Constant beta=30 seems reasonable for N=11.
        
        try:
            res_smooth = minimize(
                _objective_smooth, 
                init_params, 
                args=(30.0,),
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            current_params = res_smooth.x
            current_points = _params_to_points(current_params)
            current_min = _min_triangle_area(current_points)
            
            # Phase 2: Polish with Nelder-Mead on the true objective
            # Nelder-Mead maximizes? No, it minimizes.
            # So we minimize -min_area.
            
            res_hard = minimize(
                _objective_hard, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            final_params = res_hard.x
            final_points = _params_to_points(final_params)
            final_min = _min_triangle_area(final_points)
            
            if final_min > best_min:
                best_min = final_min
                best_points = final_points.copy()
                
        except Exception as e:
            # If L-BFGS-B fails, try Nelder-Mead directly on initial
            try:
                res_hard = minimize(
                    _objective_hard, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                final_params = res_hard.x
                final_points = _params_to_points(final_params)
                final_min = _min_triangle_area(final_points)
                
                if final_min > best_min:
                    best_min = final_min
                    best_points = final_points.copy()
            except:
                pass

    # If no valid points found, fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
# EVOLVE-BLOCK-END
