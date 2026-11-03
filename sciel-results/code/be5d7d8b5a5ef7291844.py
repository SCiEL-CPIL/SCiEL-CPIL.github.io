"""
Rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.

Strategy: Start with schoolbook (rank 105), then search for lower-rank decompositions
using iterative refinement. Return the best valid decomposition found.
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Precompute all needed indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, rank, time_limit=5.0, seed=42):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Returns (U_list, V_list, W_list) or None if not found.
    """
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Multiple restarts
    for attempt in range(10):
        if time.time() - start > time_limit:
            break
        
        np.random.seed(seed + attempt)
        
        # Initialize with random values
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Scale to get reasonable magnitudes
        scale = 1.0 / np.sqrt(rank)
        U *= scale
        V *= scale
        W *= scale
        
        # Iterative refinement
        for it in range(200):
            if time.time() - start > time_limit:
                break
            
            # Compute error and gradients
            err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        current = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                        residual = current - 1.0
                        err += residual * residual
                        
                        # Gradients
                        grad_U[idx_u] += residual * (V[idx_v] * W[idx_w])
                        grad_V[idx_v] += residual * (U[idx_u] * W[idx_w])
                        grad_W[idx_w] += residual * (U[idx_u] * V[idx_v])
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Gradient descent step with adaptive learning rate
            lr = 0.1 / (1.0 + it * 0.01)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8 and best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists of integers
    representing the factor matrices for the tensor decomposition.
    
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Convention:
    W[k*n + i, r] is the coefficient for the r-th term in the decomposition.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, n * p)
    
    # Try ranks from lower_bound up to best_rank-1
    # Focus on promising ranges
    ranks_to_try = list(range(lower_bound, lower_bound + 20)) + \
                   list(range(lower_bound + 20, min(best_rank - 1, lower_bound + 50), 5))
    
    # Remove duplicates and sort
    ranks_to_try = sorted(set(r for r in ranks_to_try if r < best_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try to find decomposition with this rank
        result = _try_find_decomposition(n, m, p, rank, time_limit=3.0, seed=42)
        
        if result is not None:
            U_list, V_list, W_list = result
            
            # Verify the result
            if _verify_np(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break  # Found a valid lower-rank decomposition
    
    return best_result
# EVOLVE-BLOCK-END
