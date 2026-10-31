import time
import numpy as np
from scipy.optimize import minimize

N = 16

def _score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    if min_d_sq <= 0:
        return np.inf
    return max_d_sq / min_d_sq

def _smooth_objective(flat_points, p):
    """
    Smoothed objective using Lp norms.
    Minimizes log(R_p) where R_p approximates max/min ratio.
    """
    points = flat_points.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, 1e-10) # Avoid log(0)
    
    # We want to minimize max/min.
    # Max approx: (sum d^p)^(1/p)
    # Min approx: (sum d^-p)^(-1/p)
    # Ratio approx: (sum d^p)^(1/p) / (sum d^-p)^(-1/p)
    # Log Ratio: 1/p * log(sum d^p) + 1/p * log(sum d^-p)
    
    # Use squared distances for stability?
    # d = sqrt(d_sq)
    # d^p = (d_sq)^(p/2)
    
    p_half = p / 2.0
    sum_max = np.sum(dists_sq ** p_half)
    sum_min = np.sum(dists_sq ** (-p_half))
    
    # Avoid log of 0 or inf
    if sum_max <= 0 or sum_min <= 0:
        return 1e10
        
    log_ratio = (1.0 / p) * (np.log(sum_max) + np.log(sum_min))
    return log_ratio

def _hex_seed():
    pts = np.array([[0.0, 0.0]])
    for i in range(6):
        angle = i * 2 * np.pi / 6
        pts = np.vstack([pts, [np.cos(angle), np.sin(angle)]])
    for i in range(12):
        angle = i * 2 * np.pi / 12
        if len(pts) < 16:
            pts = np.vstack([pts, [2*np.cos(angle), 2*np.sin(angle)]])
        if len(pts) >= 16:
            break
    return pts

def _grid_seed():
    side = 4
    return np.array([[i % side, i // side] for i in range(N)], dtype=float)

def _circle_seed():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _random_seed():
    return np.random.rand(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    seeds = [
        _hex_seed(),
        _grid_seed(),
        _circle_seed(),
        _random_seed(),
        _random_seed()
    ]
    
    best_score = np.inf
    best_points = None
    
    # Annealing schedule for p
    p_values = [4, 16, 64]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Center seed
        x0 = seed - np.mean(seed, axis=0)
        
        current_x = x0
        try:
            # Stage 1: Smooth optimization
            for p in p_values:
                if time.time() > deadline:
                    break
                res = minimize(
                    lambda x: _smooth_objective(x, p),
                    current_x.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                current_x = res.x.reshape(N, 2)
                
            # Stage 2: Exact polish with Powell
            if time.time() < deadline - 5:
                res_exact = minimize(
                    lambda x: _score(x.reshape(N, 2)),
                    current_x.flatten(),
                    method='Powell',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_exact.fun < best_score:
                    best_score = res_exact.fun
                    best_points = res_exact.x.reshape(N, 2)
                    
        except Exception:
            pass
            
    if best_points is None:
        best_points = _hex_seed()
        
    return best_points
