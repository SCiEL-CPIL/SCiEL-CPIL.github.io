import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """
    Round the entries of U, V, W to the nearest half-integer (multiple of 0.5).
    """
    def round_matrix(M):
        return [[round(x * 2) / 2 for x in row] for row in M]
    return round_matrix(U), round_matrix(V), round_matrix(W)

def _construct_schoolbook(n=3, m=3, p=3):
    """
    Construct the schoolbook rank 27 decomposition.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _search_rank23(n=3, m=3, p=3, time_limit=10):
    """
    Attempt to find a rank 23 decomposition using gradient descent.
    Returns None if not found or time exceeded.
    """
    start_time = time.time()
    rank = 23
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Target tensor T
    # T[i,j,k] = 1 if we are computing the correct product term
    # Condition: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # Note: The indices in the prompt are U[i*m+j], V[j*p+k], W[k*n+i]
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err += (s - 1.0) ** 2
        return err

    # Simple gradient descent
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradients
        # This is computationally expensive, but n,m,p are small.
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    grad = 2 * (s - 1.0)
                    for c in range(rank):
                        dU[i*m+j][c] += grad * V[j*p+k][c] * W[k*n+i][c]
                        dV[j*p+k][c] += grad * U[i*m+j][c] * W[k*n+i][c]
                        dW[k*n+i][c] += grad * U[i*m+j][c] * V[j*p+k][c]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if compute_error(U, V, W) < 1e-4:
            # Round to half integers
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded

    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a rank 23 decomposition
    result = _search_rank23(n, m, p, time_limit=10)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
