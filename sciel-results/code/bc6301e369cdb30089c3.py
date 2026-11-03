"""Fifty equal circles in a unit square – improved packing via local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _radius(pts):
    """Compute the achievable common radius for a set of centres."""
    # Wall constraints: distance to each side
    wall = 0.5 - np.abs(pts)
    wall_min = wall.min()
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min() / 2.0
    return min(wall_min, pair_min)


def _negative_radius(pts):
    """Negative of the radius (for minimization)."""
    return -_radius(pts)


def _soft_neg_radius(pts, beta=100.0):
    """
    Smooth surrogate: -softmin of (wall distances, pairwise/2).
    softmin(x) ≈ -log-sum-exp(-x) / beta, which approximates min(x) as beta→∞.
    """
    pts = np.asarray(pts).reshape(-1, 2)
    # Wall distances
    wall = 0.5 - np.abs(pts)
    wall_vals = wall.ravel()
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_vals = (dist / 2.0).ravel()
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    # softmin = -logsumexp(-beta * x) / beta
    neg_beta_x = -beta * all_vals
    max_val = neg_beta_x.max()
    logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    softmin_val = -logsumexp / beta
    return -softmin_val


def _soft_neg_radius_grad(pts, beta=100.0):
    """
    Gradient of the soft-negative-radius surrogate.
    This is expensive to compute analytically for all pairs, so we use
    numerical gradient as a fallback, but L-BFGS-B can handle that.
    """
    return None  # Let L-BFGS-B use numerical gradients


def _grid_init(n_cols, n_rows, total):
    """Generate a grid-based initial layout."""
    m = n_cols
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for y in ticks[:n_rows]:
        for x in ticks:
            pts.append([x, y])
    return np.array(pts[:total])


def _random_init(n):
    """Random initial positions inside the square."""
    return (np.random.rand(n, 2) - 0.5) * 0.9


def construct():
    t_start = time.time()
    deadline = 150.0  # seconds
    
    best_pts = None
    best_r = -np.inf
    
    # Start with a grid layout as initial guess
    m = int(np.ceil(np.sqrt(N)))
    grid_pts = _grid_init(m, m, N)
    best_pts = grid_pts.copy()
    best_r = _radius(best_pts)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    
    # First, try optimizing from the grid
    try:
        res = minimize(
            _soft_neg_radius,
            grid_pts,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
            args=(50.0,)
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            r = _radius(res.x)
            if r > best_r:
                best_r = r
                best_pts = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - t_start > deadline - 10:
        return best_pts
    
    # Try several random restarts
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() - t_start > deadline - 15:
            break
        
        # Random init
        pts = _random_init(N)
        
        # Optimize
        try:
            res = minimize(
                _soft_neg_radius,
                pts,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
                args=(50.0,)
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                r = _radius(res.x)
                if r > best_r:
                    best_r = r
                    best_pts = res.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - t_start > deadline - 10:
            break
    
    # Polish: use a higher beta for more accuracy
    if time.time() - t_start < deadline - 20:
        try:
            res = minimize(
                _soft_neg_radius,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
                args=(200.0,)
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                r = _radius(res.x)
                if r > best_r:
                    best_r = r
                    best_pts = res.x.copy()
        except Exception:
            pass
    
    # Final polish with even higher beta if time allows
    if time.time() - t_start < deadline - 10:
        try:
            res = minimize(
                _soft_neg_radius,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100},
                args=(500.0,)
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                r = _radius(res.x)
                if r > best_r:
                    best_r = r
                    best_pts = res.x.copy()
        except Exception:
            pass
    
    # Ensure all points are strictly inside the square
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_pts

# EVOLVE-BLOCK-END
