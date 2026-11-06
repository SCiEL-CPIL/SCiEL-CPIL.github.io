"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _radius(c):
    """Compute the common radius for centres c (N,2) inside [-0.5,0.5]^2."""
    if not np.all(np.isfinite(c)):
        return -1.0
    if c.shape != (N, 2):
        return -1.0
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    min_wall = np.min(wall)
    if min_wall <= 0:
        return -1.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    if min_pair <= 0:
        return -1.0
    return min(min_wall, min_pair / 2.0)


def _soft_min(values, beta):
    """Smooth approximation of min via log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _objective_soft(c, beta):
    """
    Negative of a soft-max surrogate for the radius.
    We maximize min(wall_dists, pair_dists/2).
    Equivalently minimize -min(...) using soft-min on the negatives.
    """
    c = c.reshape(N, 2)
    wall = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_half = dist / 2.0
    
    # We want to maximize min(wall, pair_half)
    # = - min(-wall, -pair_half)
    # Use soft-min on the negative values
    neg_vals = np.concatenate([-wall, -pair_half.ravel()])
    return _soft_min(neg_vals, beta)


def _objective_exact(c):
    """Exact negative radius (for final evaluation)."""
    c = c.reshape(N, 2)
    r = _radius(c)
    if r < 0:
        return 1e10
    return -r


def _initial_grid():
    """Create an initial square grid of N points in the unit square."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_perturb(c, scale=0.01):
    """Add small random perturbation."""
    return c + scale * np.random.randn(N, 2)


def construct():
    deadline = time.time() + 110  # safety margin before 120s
    
    np.random.seed(42)
    
    # Start with grid
    best_c = _initial_grid()
    best_r = _radius(best_c)
    
    # Phase 1: Perturb + optimize with moderate beta
    betas = [20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Try a few perturbations at this beta
        n_trials = 3 if beta <= 50 else 2
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            if trial == 0 and beta == betas[0]:
                c0 = best_c.copy()
            else:
                c0 = _random_perturb(best_c, scale=0.005)
            
            # Optimize using L-BFGS-B with bounds
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            result = minimize(
                _objective_soft,
                c0.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            c_new = result.x.reshape(N, 2)
            r_new = _radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    
    # Phase 2: High-precision polish with exact objective
    for beta_polish in [1000, 2000]:
        if time.time() > deadline:
            break
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        result = minimize(
            _objective_soft,
            best_c.ravel(),
            args=(beta_polish,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        c_new = result.x.reshape(N, 2)
        r_new = _radius(c_new)
        
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    
    # Final polish with exact objective (nonsmooth but precise)
    if time.time() < deadline - 5:
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        result = minimize(
            _objective_exact,
            best_c.ravel(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        c_new = result.x.reshape(N, 2)
        r_new = _radius(c_new)
        
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    
    # Ensure validity
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        return _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
