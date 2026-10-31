"""Optimized 16-point placement minimizing (max_dist/min_dist)^2."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 16
D = 2

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _pairwise_dists(pts):
    """Compute all pairwise distances for N points in 2D."""
    n = len(pts)
    d = np.zeros((n, n))
    for i in range(n):
        diff = pts[i] - pts[i + 1:]
        d[i, i + 1:] = np.sqrt(np.sum(diff ** 2, axis=1))
        d[i + 1:, i] = d[i, i + 1:]
    return d


def _score_from_dists(d):
    """Compute (max/min)^2 from distance matrix."""
    min_d = np.min(d[np.triu_indices(N, k=1)])
    max_d = np.max(d[np.triu_indices(N, k=1)])
    if min_d < 1e-12:
        return 1e12
    return (max_d / min_d) ** 2


def _objective(x):
    """Objective function: (max_dist/min_dist)^2 for 16 points in 2D."""
    pts = x.reshape(N, D)
    d = _pairwise_dists(pts)
    return _score_from_dists(d)


def _soft_objective(x, beta=1.0):
    """Smooth surrogate using log-sum-exp for max and min."""
    pts = x.reshape(N, D)
    d = _pairwise_dists(pts)
    d_upper = d[np.triu_indices(N, k=1)]
    # Soft max via log-sum-exp
    max_val = np.max(d_upper)
    soft_max = max_val + np.log(np.sum(np.exp(d_upper - max_val)))
    # Soft min via -log-sum-exp of negatives
    min_val = np.min(d_upper)
    soft_min = -(-min_val + np.log(np.sum(np.exp(-(d_upper - min_val)))))
    if soft_min < 1e-12:
        return 1e12
    return (soft_max / soft_min) ** 2


def _grid_init():
    """4x4 grid initialization."""
    side = 4
    grid = np.array(list(itertools.product(range(side), repeat=2)), dtype=float)
    return grid.flatten()


def _hex_init():
    """Hexagonal-like lattice initialization for 16 points."""
    pts = []
    # Use a hexagonal packing pattern
    for row in range(4):
        for col in range(4):
            x = col + 0.5 * (row % 2)
            y = row * 0.866  # sqrt(3)/2
            pts.append([x, y])
    return np.array(pts, dtype=float).flatten()


def _random_init():
    """Random initialization in unit square."""
    return np.random.uniform(0, 1, size=(N, D)).flatten()


def _optimize_from(x0, max_iter=500, method='L-BFGS-B'):
    """Run optimization from initial point x0."""
    try:
        result = minimize(
            _objective, x0, method=method,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        return result.x, result.fun
    except Exception:
        return x0, _objective(x0)


def _polish(x0, max_iter=1000):
    """High-precision polish pass with tighter tolerance."""
    try:
        result = minimize(
            _objective, x0, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        return result.x, result.fun
    except Exception:
        return x0, _objective(x0)


def construct():
    global START_TIME
    best_x = None
    best_score = float('inf')
    
    # Initial candidates
    inits = [
        ('grid', _grid_init()),
        ('hex', _hex_init()),
        ('random1', _random_init()),
        ('random2', _random_init()),
        ('random3', _random_init()),
    ]
    
    # Try each initialization with L-BFGS-B
    for name, x0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        x_opt, score = _optimize_from(x0, max_iter=300)
        if score < best_score:
            best_score = score
            best_x = x_opt.copy()
    
    # Polish the best solution found
    if best_x is not None and time.time() - START_TIME < TIME_LIMIT:
        x_polished, score_polished = _polish(best_x, max_iter=500)
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished
    
    # Additional random restarts if time permits
    for i in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        x0 = _random_init()
        x_opt, score = _optimize_from(x0, max_iter=200)
        if score < best_score:
            best_score = score
            best_x = x_opt.copy()
            # Polish the new best
            if time.time() - START_TIME < TIME_LIMIT - 5:
                x_polished, score_polished = _polish(best_x, max_iter=300)
                if score_polished < best_score:
                    best_score = score_polished
                    best_x = x_polished
    
    # Final polish if time permits
    if best_x is not None and time.time() - START_TIME < TIME_LIMIT - 2:
        x_final, score_final = _polish(best_x, max_iter=300)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final
    
    if best_x is None:
        # Fallback to grid
        best_x = _grid_init()
    
    return best_x.reshape(N, D)
