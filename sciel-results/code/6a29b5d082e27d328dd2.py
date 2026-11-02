"""Rewritten program for the 50-sphere packing task."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START

def _score(c):
    """Compute the packing radius for centers c (N,3)."""
    # Wall constraints
    walls = 0.5 - np.abs(c)
    min_wall = np.min(walls)
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _initial_guess(seed=42):
    """Generate an initial guess using a jittered grid."""
    rng = np.random.RandomState(seed)
    # Use a 4x4x4 grid (64 positions), pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take first N
    idx = rng.permutation(len(grid))[:N]
    pts = grid[idx].copy()
    # Add small jitter to break symmetry
    pts += rng.normal(0, 0.01, size=pts.shape)
    # Clamp to cube
    np.clip(pts, -0.49, 0.49, out=pts)
    return pts


def _objective(c):
    """Negative radius to minimize."""
    return -_score(c)


def _soft_min(x, beta=50.0):
    """Soft-min via log-sum-exp trick for negative values."""
    # soft min of x_i = - (1/beta) * log(sum(exp(-beta*x_i)))
    min_x = np.min(x)
    return -min_x - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (x - min_x))))


def _soft_objective(c, beta=50.0):
    """Smooth surrogate: maximize min of all constraints."""
    # Pairwise distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_vals = d / 2.0
    # Wall distances
    wall_vals = 0.5 - np.abs(c)
    # Combine all
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    return -_soft_min(all_vals, beta)


def _polish(c, max_iter=200, tol=1e-10):
    """Local polish using gradient-free or gradient-based method."""
    best_c = c.copy()
    best_score = _score(c)
    
    # Use Nelder-Mead with tight settings
    from scipy.optimize import minimize
    
    def obj(x):
        return -_score(x.reshape(N, 3))
    
    x0 = c.ravel()
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    
    if res.fun < -best_score:
        best_c = res.x.reshape(N, 3)
        best_score = -res.fun
    
    return best_c, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with jittered grid
    c = _initial_guess(seed=42)
    best_c = c.copy()
    best_score = _score(c)
    
    # Phase 1: Nelder-Mead optimization on the raw score
    def obj(x):
        return -_score(x.reshape(N, 3))
    
    x0 = c.ravel()
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    elapsed = time.time() - start_time
    if res.fun < -best_score:
        best_c = res.x.reshape(N, 3)
        best_score = -res.fun
    
    # Phase 2: Try soft-min objective for smoother landscape
    if time.time() < deadline - 30:
        def soft_obj(x):
            return _soft_objective(x.reshape(N, 3), beta=30.0)
        
        res2 = minimize(soft_obj, best_c.ravel(), method='L-BFGS-B',
                        bounds=[(-0.5, 0.5)] * (N * 3),
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() < deadline - 10:
            candidate = res2.x.reshape(N, 3)
            s = _score(candidate)
            if s > best_score:
                best_c = candidate
                best_score = s
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        best_c, best_score = _polish(best_c, max_iter=100, tol=1e-10)
    
    # Final check and clamp
    np.clip(best_c, -0.499999, 0.499999, out=best_c)
    
    return best_c

# EVOLVE-BLOCK-END
