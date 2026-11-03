"""Optimized circle packing in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 15.0  # seconds to keep before deadline


def _score(x):
    """Compute the packing score for a flat array of 2N coordinates."""
    c = x.reshape(N, 2)
    # Distances between all pairs
    # Use broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Distance to boundary: min over i,k of (0.5 - |c_ik|)
    boundary = 0.5 - np.abs(c)
    min_boundary = np.min(boundary)
    return min(min_dist / 2.0, min_boundary)


def _objective(x):
    """Negative score for minimization."""
    return -_score(x)


def _soft_objective(x, k=50.0):
    """
    Smooth surrogate: -softmin(min_dist/2, min_boundary)
    using log-sum-exp for the min over all constraints.
    """
    c = x.reshape(N, 2)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    # Boundary distances
    boundary = 0.5 - np.abs(c)
    min_boundary = np.min(boundary)
    # We want max(min(min_dist/2, min_boundary))
    # Use soft-min via log-sum-exp: softmin(a,b) ≈ -(1/k)*log(exp(-k*a) + exp(-k*b))
    # But we have many constraints. Let's use a simpler approach:
    # The score is min of two quantities. We can use a smooth min.
    a = min_dist
    b = min_boundary
    m = np.minimum(a, b)
    # Smooth approximation of min
    log_sum = np.log(np.exp(-k * (a - m)) + np.exp(-k * (b - m)))
    soft_min = m - np.log(log_sum) / k
    return -soft_min


def _initial_guess_grid():
    """Square grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].flatten()


def _initial_guess_honeycomb():
    """Honeycomb-like initial guess."""
    # Try a hexagonal packing arrangement
    # For 50 circles, try rows of 7,7,7,7,7,7,6,2 or similar
    # Actually, let's just use a slightly perturbed grid
    m = 8  # 8x8 grid = 64, take 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first 50
    return grid[:N].flatten()


def _initial_guess_random(seed=42):
    """Random initial guess inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2)).flatten()


def _refine(x0, max_iter=500, tol=1e-10, deadline=None):
    """
    Refine a starting point using SLSQP with bounds.
    Returns (best_x, best_score).
    """
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x)
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    start_time = time.time()
    
    # First pass: standard SLSQP
    try:
        res = minimize(
            _objective, x, method='SLSQP', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'iprint': 0}
        )
        if res.success or res.fun < -best_score:
            if _score(res.x) > best_score:
                best_score = _score(res.x)
                best_x = res.x.copy()
    except Exception:
        pass
    
    if deadline is not None:
        if time.time() - start_time > deadline - SAFETY_MARGIN:
            return best_x, best_score
    
    # Second pass: refine from the best point with tighter tolerance
    try:
        res2 = minimize(
            _objective, best_x, method='SLSQP', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'iprint': 0}
        )
        if res2.success or res2.fun < -best_score:
            if _score(res2.x) > best_score:
                best_score = _score(res2.x)
                best_x = res2.x.copy()
    except Exception:
        pass
    
    return best_x, best_score


def _refine_soft(x0, k_start=10.0, k_end=200.0, n_steps=5, max_iter=300, deadline=None):
    """
    Progressive soft-min refinement: start with small k (smooth) and increase.
    """
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x)
    bounds = [(-0.5, 0.5)] * (2 * N)
    start_time = time.time()
    
    for i in range(n_steps):
        k = k_start * (k_end / k_start) ** (i / max(n_steps - 1, 1))
        try:
            res = minimize(
                lambda xx: _soft_objective(xx, k=k), x,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'iprint': 0}
            )
            if res.success:
                s = _score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
                x = res.x.copy()
        except Exception:
            break
        
        if deadline is not None:
            if time.time() - start_time > deadline - SAFETY_MARGIN:
                break
    
    # Final SLSQP polish
    try:
        res = minimize(
            _objective, best_x, method='SLSQP', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'iprint': 0}
        )
        if res.success or res.fun < -best_score:
            s = _score(res.x)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_score = -np.inf
    
    # Generate multiple initial guesses
    guesses = []
    
    # 1. Grid
    guesses.append(_initial_guess_grid())
    
    # 2. Perturbed grid with different seeds
    for seed in [1, 2, 3, 7, 13, 42, 100, 123]:
        base = _initial_guess_grid()
        rng = np.random.RandomState(seed)
        perturbed = base + rng.uniform(-0.01, 0.01, size=base.shape)
        # Clip to bounds
        perturbed = np.clip(perturbed, -0.49, 0.49)
        guesses.append(perturbed)
    
    # 3. Random guesses
    for seed in [10, 20, 30, 50, 77, 99]:
        guesses.append(_initial_guess_random(seed=seed))
    
    # Refine each guess
    for i, x0 in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Use soft refinement for first few, direct SLSQP for others
        if i < 4:
            x, s = _refine_soft(x0, deadline=deadline)
        else:
            x, s = _refine(x0, deadline=deadline)
        
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline:
        x, s = _refine(best_x, max_iter=1000, tol=1e-13, deadline=deadline)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    if best_x is None:
        # Fallback: return grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
