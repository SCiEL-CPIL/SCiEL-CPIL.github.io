import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    T = _build_tensor(n, m, p)
    # T_calc[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - T[i, j, k]) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce the rank by finding linear combinations of rank-1 terms."""
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to eliminate columns by checking if they can be expressed as linear combinations
    # This is a simplified approach - in practice, we'd need more sophisticated methods
    
    # For now, just return the current decomposition if we can't reduce
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, n*p) = 21
    rank_lb = max(n * m, m * p, n * p)
    
    # For this problem, try ranks from 21 up to 62
    # Use a structured search approach
    for R in range(rank_lb, best_rank):
        if time.time() > deadline:
            break
        
        # Try random initialization with structured patterns
        found = False
        num_restarts = 5 if R <= 30 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 100 + restart)
            
            # Initialize with small random values
            U_init = np.random.randint(-1, 2, size=(n * m, R)).astype(float)
            V_init = np.random.randint(-1, 2, size=(m * p, R)).astype(float)
            W_init = np.random.randint(-1, 2, size=(p * n, R)).astype(float)
            
            # Quick check
            if _verify_decomposition(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p, tol=1e-10):
                best_rank = R
                best_U = U_init.tolist()
                best_V = V_init.tolist()
                best_W = W_init.tolist()
                found = True
                break
        
        if found:
            break
    
    # Return the best result found
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
