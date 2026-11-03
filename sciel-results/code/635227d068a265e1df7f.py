"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
# i < j < k
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Uses the mapping:
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    
    # Ensure bounds are respected (though optimizer should keep them there)
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
    x = lam1 + 0.5 * lam2
    y = (sqrt(3)/2) * lam2
    => lam2 = 2 * y / sqrt(3)
    => lam1 = x - 0.5 * lam2 = x - y / sqrt(3)
    
    s = lam1 + lam2
    t = lam2 / s  (if s > 0)
    
    Note: This is the inverse of _param_to_points.
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

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        # Generate random candidates
        # Uniform sampling in triangle:
        # r1, r2 in [0,1]
        # if r1 + r2 > 1, r1 = 1-r1, r2 = 1-r2
        # point = r1*V1 + r2*V2
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
        # dist[c, k] = ||candidates[c] - points[k]||
        # We want to maximize the minimum distance.
        
        # Efficient calculation:
        # diff shape: (n_candidates, k, 2)
        # dist shape: (n_candidates, k)
        # min_dist shape: (n_candidates,)
        
        # To save memory/time, we can do this in chunks or just use broadcasting if n_candidates is small.
        # n_candidates=200, k<=10. 200*10*2 = 4000 floats. Very small.
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initializations using FPS
    n_initializations = 20
    initial_params = []
    
    for i in range(n_initializations):
        if time_left() < 10.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Also add a few random initializations to ensure diversity
    for i in range(5):
        if time_left() < 10.0:
            break
        params = rng.uniform(0.1, 0.9, size=2*N)
        initial_params.append(params)
        
    # 2. Local Polish with Nelder-Mead
    # We refine all generated candidates.
    # We sort them by their initial score to prioritize better starts?
    # Or just run them all. Nelder-Mead is fast enough for 22 dims with ~1000-2000 iterations.
    
    for start_p in initial_params:
        if time_left() < 5.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    return np.array(points)
# EVOLVE-BLOCK-END
