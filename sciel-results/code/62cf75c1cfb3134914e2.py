"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START

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

def _validate(U, V, W):
    """Check if U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the matmul tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Equivalently, for each (i,k), the matrix M_ik = sum_r W[k*n+i,r] * (U_i * V_k) must equal identity-like
    # where U_i is row i*m:(i+1)*m of U, V_k is row k*p:(k+1)*p of V
    
    # Actually the tensor T has T[i,j,k] = 1 for all valid i,j,k (since matmul tensor is all-ones in this indexing)
    # Wait, no. The matmul tensor for C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by computing the left side for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, timeout=10.0):
    """Try to find a rank-r decomposition using random search."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    for trial in range(20):
        if time.time() - start > timeout:
            break
        # Random initialization with small integers/half-integers
        U = np.random.randint(-2, 3, (n*m, r)) * 0.5
        V = np.random.randint(-2, 3, (m*p, r)) * 0.5
        W = np.random.randint(-2, 3, (p*n, r)) * 0.5
        
        # Check
        if _validate(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook (rank 75) as guaranteed fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try lower ranks with random search
    for r in range(70, 75):
        if time.time() > deadline:
            break
        result = _try_rank(r, timeout=15.0)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
# EVOLVE-BLOCK-END
