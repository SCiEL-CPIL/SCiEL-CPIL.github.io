# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _score(points):
    """Compute R = (d_min/d_max)^2 from 16 points."""
    d = pdist(points)
    if d.size == 0:
        return 0.0
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_score(points, tau):
    """Smooth surrogate: soft-min/soft-max of squared distances."""
    d2 = pdist(points) ** 2
    if d2.size == 0:
        return 0.0
    # soft max of d2
    m = d2.max()
    logsumexp_max = m + np.log(np.sum(np.exp(d2 - m)))
    # soft min of d2 = -soft max of -d2
    m2 = (-d2).max()
    logsumexp_min_neg = m2 + np.log(np.sum(np.exp(-d2 - m2)))
    soft_min = -logsumexp_min_neg
    soft_max = logsumexp_max
    if soft_max <= 0:
        return 0.0
    return (soft_min / soft_max) ** 2

def _negative_smooth_score(params, tau):
    """Objective for minimization: -smooth_score."""
    n = 16
    points = params.reshape(n, 2)
    return -_smooth_score(points, tau)

def _negative_score(params, tau):
    """Objective for minimization: -score (nonsmooth)."""
    n = 16
    points = params.reshape(n, 2)
    return -_score(points)

def _normalize(points):
    """Fix centroid to 0 and scale so that sum of squared norms = n."""
    points = points - np.mean(points, axis=0)
    ss = np.sum(points ** 2)
    if ss > 1e-15:
        points = points * np.sqrt(16.0 / ss)
    return points

def _grid_init():
    """4x4 grid centered at origin, normalized."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return _normalize(points)

def _random_init(seed):
    """Random points, normalized."""
    rng = np.random.RandomState(seed)
    points = rng.randn(16, 2)
    return _normalize(points)

def _optimize_from(points0, deadline, tau_init=1.0):
    """Run smooth SQP from points0, then polish with decreasing tau.
    Returns best points and score found."""
    n = 16
    best_points = points0.copy()
    best_score = _score(best_points)
    
    # Multi-round with decreasing tau for smoothness
    taus = [tau_init, 0.5, 0.1, 0.01, 0.001]
    
    for tau in taus:
        if time.time() > deadline:
            break
        params0 = best_points.ravel()
        try:
            res = minimize(
                _negative_smooth_score,
                params0,
                args=(tau,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            if res.success or res.fun < -best_score:
                cand = _normalize(res.x.reshape(n, 2))
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand
        except Exception:
            pass
    
    # Final high-precision polish with nonsmooth objective
    if time.time() < deadline:
        for tol in [1e-7, 1e-9, 1e-11]:
            if time.time() > deadline:
                break
            params0 = best_points.ravel()
            try:
                res = minimize(
                    _negative_score,
                    params0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': tol, 'fatol': tol}
                )
                cand = _normalize(res.x.reshape(n, 2))
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand
            except Exception:
                pass
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before 630s kill
    
    best_points = _grid_init()
    best_score = _score(best_points)
    
    # Multiple restarts: grid + random
    inits = [best_points]
    for seed in range(40):
        inits.append(_random_init(seed))
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        # Reserve time for remaining inits
        remaining_inits = len(inits) - i
        time_per_init = (deadline - time.time()) / max(remaining_inits, 1)
        if time_per_init < 5.0:
            break
        local_deadline = time.time() + min(time_per_init, 30.0)
        
        pts, sc = _optimize_from(init.copy(), local_deadline)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Final polish on best with remaining time
    if time.time() < deadline:
        pts, sc = _optimize_from(best_points.copy(), deadline, tau_init=0.01)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Ensure valid output
    best_points = _normalize(best_points)
    assert best_points.shape == (16, 2)
    assert np.all(np.isfinite(best_points))
    
    return best_points

# EVOLVE-BLOCK-END
