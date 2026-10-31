"""Improved program: Global search with Differential Evolution followed by local refinement."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DEADLINE = 110.0  # safety margin before 120s limit


def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances, shape (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _score(P):
    """(max pairwise dist / min pairwise dist)^2 over distinct pairs."""
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    vmin = vals.min()
    vmax = vals.max()
    if vmin <= 1e-15:
        return 1e12
    return (vmax / vmin)


def _smooth_objective(P, beta=50.0):
    """
    Smooth surrogate for local refinement.
    Minimizes the soft range of squared distances.
    """
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    
    # Soft max and soft min of squared distances
    # Using log-sum-exp trick for stability
    m = vals.max()
    logsumexp_max = m + np.log(np.sum(np.exp(vals - m)))
    
    mn = vals.min()
    logsumexp_min_neg = -mn + np.log(np.sum(np.exp(-vals + mn)))
    soft_min = -logsumexp_min_neg
    
    # We want to minimize the ratio, which is equivalent to minimizing the spread.
    # A good proxy is (soft_max - soft_min) / soft_min
    if soft_min <= 1e-8:
        return 1e6
    return (logsumexp_max - soft_min) / soft_min


def _raw_objective(P):
    """
    Raw objective for global search (Differential Evolution).
    Returns the squared ratio.
    """
    return _score(P)


def construct():
    t_start = time.time()
    
    # 1. Global Search using Differential Evolution
    # Bounds: [-10, 10] for each coordinate
    bounds = [(-10, 10)] * (N * D)
    
    # Time budget for DE: 80 seconds
    # DE is the primary engine. We allocate most of the time here.
    
    try:
        # Differential Evolution parameters
        # popsize: 15 * 32 = 480 individuals.
        # maxiter: 100 iterations.
        # Total evaluations ~ 48,000. Fast for N=16.
        
        res_de = differential_evolution(
            _raw_objective,
            bounds,
            popsize=15,
            maxiter=100,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polish
            init='latinhypercube'
        )
        
        if res_de.success or res_de.fun < 1e10:
            P_best = res_de.x.reshape(N, D)
        else:
            # Fallback to grid if DE fails
            P_best = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
            
    except Exception:
        P_best = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
        
    elapsed_de = time.time() - t_start
    
    # 2. Local Refinement using L-BFGS-B
    # Time budget for refinement: remaining time - 10s safety
    time_left = DEADLINE - 10 - elapsed_de
    if time_left > 5:
        try:
            res_local = minimize(
                _smooth_objective,
                P_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            P_refined = res_local.x.reshape(N, D)
            # Check if refined is better
            if _score(P_refined) < _score(P_best):
                P_best = P_refined
        except Exception:
            pass
            
    # 3. Final sanity check and distinctness
    D2 = _pairwise_dists_sq(P_best)
    iu = np.triu_indices(N, k=1)
    if D2[iu].min() < 1e-10:
        # Perturb
        np.random.seed(42)
        P_best += np.random.randn(N, D) * 1e-5

    return P_best
# EVOLVE-BLOCK-END
