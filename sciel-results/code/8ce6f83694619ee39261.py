import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle in Cartesian
# Area of this triangle is sqrt(3)/4. 
# The problem statement says "unit-area triangle" but the interface specifies 
# (0,0), (1,0), (1/2, sqrt(3)/2). 
# The area of this specific triangle is sqrt(3)/4.
# The evaluator measures the raw value. The seed program divides by area? 
# "The notebook uses an equilateral triangle and divides by its area."
# However, the interface says: "construct() -> array ... The evaluator measures the raw value".
# Let's look at the current program. It returns points. 
# The score is "value/seed". 
# If the evaluator calculates the min area and divides by the triangle area, 
# then maximizing the min area is equivalent to maximizing the normalized min area.
# We just need to maximize the raw min area of the triangles formed by the points.

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    
    Point = lambda_0 * V0 + lambda_1 * V1 + lambda_2 * V2
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    
    # Ensure bounds are respected strictly to avoid numerical issues
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    # Cartesian coordinates
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    Point = lam1 * V1 + lam2 * V2
    x = lam1 * 1 + lam2 * 0.5
    y = lam1 * 0 + lam2 * (sqrt(3)/2)
    
    => lam2 = 2 * y / sqrt(3)
    => lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = lam2 / s  (if s > 0, else 0)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    # Clip to valid range [0, 1]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        # Generate random candidates uniformly in the triangle
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        # Calculate min distance from each candidate to existing points
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    # FPS initializations are generally very good for packing problems
    n_fps_inits = 15
    for i in range(n_fps_inits):
        if time_left() < 30.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add a few random initializations to ensure diversity
    for i in range(10):
        if time_left() < 30.0:
            break
        # s in [0.1, 0.9], t in [0.1, 0.9] to avoid corners initially
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP
    # SLSQP is good for constrained problems, though our mapping handles constraints.
    # We use bounds [0,1] on s and t.
    
    for start_p in initial_params:
        if time_left() < 15.0:
            break
            
        try:
            # First pass with SLSQP
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Polish the best solution found with Nelder-Mead for finer tuning
    # Nelder-Mead can refine the solution further, especially near the optimum
    # where the min function is non-differentiable.
    if best_params is not None and time_left() > 20.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 4. If time permits, try a few more SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                # Add small noise to best_params
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback: Center points
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    return np.array(points)
