"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _construct_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _flatten_factors(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_factors(x, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = x[:size_u]
    V_flat = x[size_u:size_u + size_v]
    W_flat = x[size_u + size_v:size_u + size_v + size_w]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    return U, V, W

def _objective(x, T, n, m, p, rank):
    U, V, W = _unflatten_factors(x, n, m, p, rank)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err

def _try_rank_decomposition(T, target_rank, num_restarts=5, max_iter=1000, tol=1e-10):
    n, m, p = DIMENSIONS
    shape = T.shape
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    best_err = np.inf
    best_x = None
    
    for trial in range(num_restarts):
        np.random.seed(trial * 7919 + target_rank)
        # Initialize with small random values
        x0 = np.random.randn(total_size) * 0.1
        
        # Use Nelder-Mead for robustness, or L-BFGS-B for speed
        # Nelder-Mead can be slow for high dimensions, but L-BFGS-B is better for smooth objectives
        # The objective is smooth (quadratic in factors). L-BFGS-B is a good choice.
        res = minimize(_objective, x0, args=(T, n, m, p, target_rank), 
                       method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    
    if best_x is not None and best_err < tol:
        U, V, W = _unflatten_factors(best_x, n, m, p, target_rank)
        return U, V, W, best_err
    else:
        return None, None, None, best_err

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    T = _construct_tensor()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    
    # Try to find a lower rank decomposition
    # Known lower bounds for 3x3x7 are not as established as 3x3x3.
    # Schoolbook is 63. Let's try from 30 upwards to find the minimum.
    # If we find rank R, we can stop.
    
    # Heuristic: start from a reasonable lower bound.
    # For 3x3x3, Strassen is 7.
    # For 3x3x7, it's likely higher. Let's try 20 to 40.
    
    for target_rank in range(20, 64):
        if time.time() > deadline:
            break
            
        # Try to find decomposition for this rank
        U, V, W, err = _try_rank_decomposition(T, target_rank, num_restarts=3, max_iter=200, tol=1e-10)
        
        if U is not None and err < 1e-10:
            # Verify strictly
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Found minimal rank (since we iterate upwards)
                
    return best
# EVOLVE-BLOCK-END
