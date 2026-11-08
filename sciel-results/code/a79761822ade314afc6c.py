"""Matrix multiplication 3x3x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 120.0

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] = 1."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the multiplication tensor.
    
    U: shape (n*m, rank) - factors for A
    V: shape (m*p, rank) - factors for B  
    W: shape (p*n, rank) - factors for C (output)
    
    The decomposition requires: for all i,j,k,
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False, s, (i, j, k)
    return True, 0.0, None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, max_rank, time_budget):
    """Try to find a decomposition of rank <= max_rank using iterative refinement."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Initialize with random rank-r decomposition
    best_result = None
    best_rank = max_rank + 1
    
    for target_rank in range(9, min(max_rank + 1, 30)):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > time_budget - 5:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative refinement: minimize sum of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            for iteration in range(200):
                if time.time() - start_time > time_budget - 5:
                    break
                
                # Compute current errors
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i, j, k] = sum(
                                U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
                                for r in range(target_rank)
                            ) - 1.0
                
                err_norm = np.linalg.norm(errors)
                if err_norm < 1e-10:
                    # Found exact decomposition!
                    result = {
                        "U": [[float(U[i*m+j, r]) for r in range(target_rank)] for i in range(n) for j in range(m)],
                        "V": [[float(V[j*p+k, r]) for r in range(target_rank)] for j in range(m) for k in range(p)],
                        "W": [[float(W[k*n+i, r]) for r in range(target_rank)] for k in range(p) for i in range(n)]
                    }
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = result
                    return best_result
                
                # Update each factor in turn
                for factor in ['U', 'V', 'W']:
                    if factor == 'U':
                        for i in range(n):
                            for j in range(m):
                                for r in range(target_rank):
                                    # Gradient with respect to U[i*m+j, r]
                                    grad = 0.0
                                    for k in range(p):
                                        grad += errors[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                                    U[i*m+j, r] -= 0.01 * grad
                    elif factor == 'V':
                        for j in range(m):
                            for k in range(p):
                                for r in range(target_rank):
                                    grad = 0.0
                                    for i in range(n):
                                        grad += errors[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                                    V[j*p+k, r] -= 0.01 * grad
                    else:  # W
                        for k in range(p):
                            for i in range(n):
                                for r in range(target_rank):
                                    grad = 0.0
                                    for j in range(m):
                                        grad += errors[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                                    W[k*n+i, r] -= 0.01 * grad
            
            # Check final error
            final_err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        final_err[i, j, k] = sum(
                            U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
                            for r in range(target_rank)
                        ) - 1.0
            if np.linalg.norm(final_err) < 1e-8:
                result = {
                    "U": [[float(U[i*m+j, r]) for r in range(target_rank)] for i in range(n) for j in range(m)],
                    "V": [[float(V[j*p+k, r]) for r in range(target_rank)] for j in range(m) for k in range(p)],
                    "W": [[float(W[k*n+i, r]) for r in range(target_rank)] for k in range(p) for i in range(n)]
                }
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                return best_result
    
    return best_result

def construct():
    """
    Return a low-rank exact decomposition of the 3x3 * 3x6 matrix multiplication tensor.
    
    Uses iterative optimization to find a rank-minimal decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a low-rank decomposition
    result = _try_rank_search(n, m, p, max_rank=25, time_budget=TIME_LIMIT - 10)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
