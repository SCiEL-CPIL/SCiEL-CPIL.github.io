import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_distances(X):
    # X is (N, D)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return dists

def _objective(X_flat):
    X = X_flat.reshape(N, D)
    dists = _pairwise_distances(X)
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    
    d_min = np.min(d)
    d_max = np.max(d)
    
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)**2

def construct():
    start_time = time.time()
    # Use 110 seconds as deadline to be safe
    deadline = start_time + 110.0
    
    best_score = float('inf')
    best_X = None
    
    # Pre-compute indices for speed in objective if needed, 
    # but _pairwise_distances is vectorized and fast enough for N=14.
    
    # Strategy: Multi-start local optimization
    # 1. Random cube starts
    # 2. Spherical starts
    
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Choose initialization type
        if i % 2 == 0:
            # Random in cube
            X0 = np.random.rand(N, D)
            X0 -= np.mean(X0, axis=0)
        else:
            # Random on sphere
            # Generate random normal vectors and normalize
            X0 = np.random.randn(N, D)
            norms = np.linalg.norm(X0, axis=1)
            X0 /= norms[:, np.newaxis]
            
        # Scale to unit max distance to normalize the problem scale
        dists0 = _pairwise_distances(X0)
        d_max0 = np.max(dists0)
        if d_max0 > 1e-6:
            X0 = X0 / d_max0
        else:
            continue

        # Optimize
        # Powell is robust for this. Nelder-Mead can be faster but more sensitive.
        # Let's try Powell with a reasonable maxiter.
        try:
            res = minimize(_objective, X0.flatten(), method='Powell', 
                           options={'maxiter': 100, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_X = res.x.reshape(N, D)
        except Exception:
            continue

    # If we have time, do a final polish with Nelder-Mead on the best solution
    if best_X is not None and time.time() < deadline - 10:
        try:
            res2 = minimize(_objective, best_X.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_X = res2.x.reshape(N, D)
        except Exception:
            pass

    # Fallback
    if best_X is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    return best_X
