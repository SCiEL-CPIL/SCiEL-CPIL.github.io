"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    # Check the tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _random_search(rank, n, m, p, max_iters=2000, seed=None):
    """Try to find a valid decomposition of given rank via random search."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for it in range(max_iters):
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-9:
            return best_U, best_V, best_W
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with schoolbook rank 36
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
    
    # Try to reduce rank
    for r in range(35, 20, -1):
        if time.time() > deadline:
            break
        result = _random_search(r, n, m, p, max_iters=500, seed=r)
        if result is not None:
            U, V, W = result
            rank = r
            if time.time() > deadline:
                break
    
    # Convert to lists
    U_list = U.tolist() if isinstance(U, np.ndarray) else U
    V_list = V.tolist() if isinstance(V, np.ndarray) else V
    W_list = W.tolist() if isinstance(W, np.ndarray) else W
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
