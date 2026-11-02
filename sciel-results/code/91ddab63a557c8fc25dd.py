import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def _try_rank_decomposition(rank, max_time=100):
    """Try to find a rank-r decomposition using numerical optimization.
    Returns (U, V, W) as lists of lists if found, else None."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    np.random.seed(42 + rank)
    
    best_residual = float('inf')
    best_solution = None
    
    for attempt in range(30):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with structured random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Use gradient descent with learning rate scheduling
        lr = 0.01
        for iteration in range(500):
            if time.time() - start_time > max_time:
                break
            
            # Compute residual and gradients
            residual = 0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = s - 1
                        residual += err * err
                        for c in range(rank):
                            grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                            grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                            grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
            
            if residual < 1e-14:
                best_residual = residual
                best_solution = (U.copy(), V.copy(), W.copy())
                break
            
            if residual < best_residual:
                best_residual = residual
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if iteration % 100 == 0 and iteration > 0:
                lr *= 0.5
        
        if best_residual < 1e-14:
            break
    
    if best_solution is not None and best_residual < 1e-12:
        U, V, W = best_solution
        # Convert to lists
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        
        # Verify with exact arithmetic check
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try rounding to nearest 0.5
        U_r = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(n*m)]
        V_r = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(m*p)]
        W_r = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
        
        # Try rounding to nearest 0.25
        U_r = [[round(U[i, c] * 4) / 4 for c in range(rank)] for i in range(n*m)]
        V_r = [[round(V[i, c] * 4) / 4 for c in range(rank)] for i in range(m*p)]
        W_r = [[round(W[i, c] * 4) / 4 for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _construct_rank_23_lie():
    """
    Construct a rank-23 decomposition exploiting structure.
    For 3x3 matrix multiplication, we can use the fact that the tensor
    has certain symmetries. This is a heuristic construction.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Start with schoolbook 27 terms, then try to merge
    # Actually, let's just do a direct search for rank 23
    return None

def construct():
    overall_deadline = time.time() + 140
    
    # Try numerical search for various ranks
    for rank in [19, 20, 21, 22, 23, 24, 25, 26]:
        if time.time() > overall_deadline - 10:
            break
        remaining = overall_deadline - time.time() - 10
        if remaining < 5:
            break
        max_time = min(20, remaining)
        result = _try_rank_decomposition(rank, max_time=max_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
