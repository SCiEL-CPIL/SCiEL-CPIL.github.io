import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns negative score (for minimization).
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1.0
    except Exception:
        return 1.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    abs_cross = np.abs(cross)
    
    # Soft min via log-sum-exp for numerical stability
    max_val = np.max(abs_cross)
    exp_vals = np.exp(-beta * (abs_cross - max_val))
    soft_min_val = max_val + np.log(np.sum(exp_vals)) / beta
    
    return -0.5 * soft_min_val / hull_area

def _make_ring_points(n_out, n_in, r, theta):
    """Create a two-ring point configuration."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _make_grid_points(n1, n2, offset=0.0):
    """Create a grid-like configuration."""
    # Place points in a roughly uniform grid
    cols = 4
    rows = 4
    points = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= N:
                break
            x = (col / (cols - 1) - 0.5) * 2.0 + offset
            y = (row / (rows - 1) - 0.5) * 2.0
            points.append([x, y])
            idx += 1
    return np.array(points[:N])

def _make_random_points(rng, scale=1.0):
    """Generate random points in a square."""
    return (rng.random((N, 2)) - 0.5) * 2.0 * scale

def _optimize_full_points(points_init, time_limit, beta=100.0):
    """
    Optimize all 28 point coordinates using L-BFGS-B on soft-min surrogate,
    then Nelder-Mead on exact objective.
    """
    start_time = time.time()
    x0 = points_init.flatten()
    
    # Phase 1: L-BFGS-B on smooth surrogate
    best_score = _calculate_score(points_init)
    best_points = points_init.copy()
    
    try:
        res = minimize(
            lambda x: _soft_min_score(x.reshape(N, 2), beta),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res.x is not None:
            points_lbfgs = res.x.reshape(N, 2)
            score_lbfgs = _calculate_score(points_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_points = points_lbfgs
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.8:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Ring configurations (7-7, 8-6, 6-8, 9-5, 5-9)
    for n_out, n_in in [(7, 7), (8, 6), (6, 8), (9, 5), (5, 9)]:
        for r in [0.3, 0.5, 0.7]:
            for theta in [0.0, np.pi/7, np.pi/6, np.pi/5]:
                pts = _make_ring_points(n_out, n_in, r, theta)
                initial_configs.append(pts)
    
    # Grid configurations
    for offset in [0.0, 0.1, -0.1]:
        pts = _make_grid_points(4, 4, offset)
        initial_configs.append(pts)
    
    # Random configurations
    for scale in [0.8, 1.0, 1.2]:
        pts = _make_random_points(rng, scale)
        initial_configs.append(pts)
    
    # Sort by initial score to prioritize good starts
    scored_configs = []
    for pts in initial_configs:
        s = _calculate_score(pts)
        scored_configs.append((s, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Optimize top configurations
    n_to_try = min(12, len(scored_configs))
    time_per_config = 10.0  # ~10s per config
    
    for i in range(n_to_try):
        if time.time() > deadline - 5.0:
            break
        
        s_init, pts_init = scored_configs[i]
        if s_init < 0.5 * best_overall_score and i > 3:
            continue  # Skip if much worse than current best
        
        t_limit = min(time_per_config, deadline - time.time() - 3.0)
        if t_limit < 3.0:
            break
        
        score, points = _optimize_full_points(pts_init, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
    
    # If we still have time, do a final polish on the best
    if best_overall_points is not None and time.time() < deadline - 10.0:
        t_remain = min(20.0, deadline - time.time() - 3.0)
        if t_remain > 5.0:
            score_final, points_final = _optimize_full_points(
                best_overall_points, t_remain, beta=200.0
            )
            if score_final > best_overall_score:
                best_overall_score = score_final
                best_overall_points = points_final
    
    # Fallback: ensure we have a valid result
    if best_overall_points is None:
        best_overall_points = _make_ring_points(7, 7, 0.5, 0.0)
        best_overall_score = _calculate_score(best_overall_points)
    
    return best_overall_points
