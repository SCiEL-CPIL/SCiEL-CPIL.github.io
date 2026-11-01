import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pair indices
pair_indices = list(itertools.combinations(range(N), 2))
M = len(pair_indices)

def _compute_distances_sq(X):
    """
    Compute squared pairwise distances.
    X: (N, D)
    Returns: (M,) array of squared distances.
    """
    # X[:, np.newaxis, :] - X[np.newaxis, :, :] -> (N, N, D)
    # dist_sq = np.sum(diff**2, axis=2)
    # We only need upper triangle.
    # For efficiency, we can use broadcasting on the flattened pair list.
    i, j = zip(*pair_indices)
    # i, j are tuples of length M
    i = np.array(i)
    j = np.array(j)
    diff = X[i] - X[j]
    dist_sq = np.sum(diff**2, axis=1)
    return dist_sq, i, j

def _objective_and_grad(x_flat, tau):
    """
    x_flat: (3*N,)
    tau: temperature
    Returns: objective value, gradient
    """
    X = x_flat.reshape(N, D)
    # Center of mass
    X = X - np.mean(X, axis=0, keepdims=True)
    
    dist_sq, i, j = _compute_distances_sq(X)
    
    # Soft max
    # Avoid overflow: shift by max
    max_val = np.max(dist_sq)
    exp_max = np.exp(tau * (dist_sq - max_val))
    Z_max = np.sum(exp_max)
    S_max = (1.0 / tau) * (np.log(Z_max) + tau * max_val)
    # p_max weights
    p_max = exp_max / Z_max
    
    # Soft min
    # min is -max of -dist_sq
    min_val = np.min(dist_sq)
    # Use -dist_sq for exp
    # To avoid underflow, shift by min_val (which is the max of -dist_sq)
    # exp(-tau * (dist_sq - min_val)) = exp(-tau * dist_sq + tau * min_val)
    exp_min = np.exp(-tau * (dist_sq - min_val))
    Z_min = np.sum(exp_min)
    S_min = -(1.0 / tau) * (np.log(Z_min) - tau * min_val)
    # p_min weights
    p_min = exp_min / Z_min
    
    # Objective: log(S_max) - log(S_min)
    if S_min <= 0 or S_max <= 0:
        return 1e10, np.zeros_like(x_flat)
        
    obj = np.log(S_max) - np.log(S_min)
    
    # Gradient
    # dS_max/dx = sum p_max_k * d(dist_sq_k)/dx
    # d(dist_sq_k)/dx_i = 2 * (x_i - x_j)
    # d(dist_sq_k)/dx_j = 2 * (x_j - x_i)
    
    grad_S_max = np.zeros_like(X)
    grad_S_min = np.zeros_like(X)
    
    # Vectorized gradient accumulation
    # diff = X[i] - X[j]
    diff = X[i] - X[j]
    
    # dS_max/dx_i += 2 * p_max * diff
    # dS_max/dx_j += 2 * p_max * (-diff)
    np.add.at(grad_S_max, i, 2.0 * p_max[:, np.newaxis] * diff)
    np.add.at(grad_S_max, j, -2.0 * p_max[:, np.newaxis] * diff)
    
    np.add.at(grad_S_min, i, 2.0 * p_min[:, np.newaxis] * diff)
    np.add.at(grad_S_min, j, -2.0 * p_min[:, np.newaxis] * diff)
    
    # dJ/dx = grad_S_max / S_max - grad_S_min / S_min
    grad_J = grad_S_max / S_max - grad_S_min / S_min
    
    return obj, grad_J.flatten()

def _exact_score(X):
    dist_sq, _, _ = _compute_distances_sq(X)
    d_min = np.min(dist_sq)
    d_max = np.max(dist_sq)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = np.inf
    best_X = None
    
    # Initializations
    # 1. Random in sphere
    # 2. Random in cube
    # 3. Perturbed grid
    
    num_restarts = 50 # We will break based on time
    
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if restart % 2 == 0:
            # Random in unit ball
            X = rng.normal(size=(N, D))
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms * rng.uniform(0.1, 1.0, size=(N, 1))
        else:
            # Random in cube [-1, 1]
            X = rng.uniform(-1, 1, size=(N, D))
            
        # Center
        X -= np.mean(X, axis=0, keepdims=True)
        
        x_flat = X.flatten()
        
        # Progressive tau
        taus = [10, 50, 200, 1000]
        
        for tau in taus:
            if time.time() > deadline:
                break
                
            # L-BFGS-B
            # maxiter limited to prevent long hangs on single run
            # maxfun limited
            try:
                res = minimize(
                    _objective_and_grad, 
                    x_flat, 
                    args=(tau,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'maxfun': 2000}
                )
                x_flat = res.x
            except Exception:
                pass
        
        # Polish with Nelder-Mead on exact objective
        # Objective for NM: exact ratio squared
        def nm_obj(x_flat):
            X = x_flat.reshape(N, D)
            X = X - np.mean(X, axis=0, keepdims=True)
            return _exact_score(X)
            
        if time.time() < deadline - 5.0: # Need time for NM
            try:
                res_nm = minimize(
                    nm_obj, 
                    x_flat, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                x_flat = res_nm.x
            except Exception:
                pass
        
        # Final evaluation
        X_final = x_flat.reshape(N, D)
        X_final -= np.mean(X_final, axis=0, keepdims=True)
        
        # Check distinctness
        dist_sq, _, _ = _compute_distances_sq(X_final)
        if np.min(dist_sq) < 1e-8:
            continue
            
        score = _exact_score(X_final)
        
        if score < best_score:
            best_score = score
            best_X = X_final.copy()
            
    if best_X is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    return best_X
