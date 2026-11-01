"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, target_rank, time_limit=10):
    """Try to find a decomposition of the given rank using random initialization and optimization.
    
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # We need to minimize sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
    # This is a non-convex problem. Let's try a simple gradient descent approach.
    
    # If target_rank >= n*m*p, just return schoolbook
    if target_rank >= n * m * p:
        return _schoolbook(n, m, p)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple iterative refinement
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        # Gradient of sum_k (sum_r' U[i*m+j,r'] V[j*p+k,r'] W[k*n+i,r'] - 1)^2 wrt U[i*m+j,r]
                        grad = 0
                        for k in range(p):
                            inner = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            grad += 2 * (inner - 1) * V[j*p+k][r] * W[k*n+i][r]
                        U[i*m+j][r] -= 0.01 * grad
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(n, m, p, target_rank, time_limit=10)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
