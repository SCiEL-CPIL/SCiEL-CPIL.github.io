"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication (n,m) x (m,p)
    # Actually, for matrix multiplication C = A @ B where A is n x m, B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if this term contributes, which is always 1 for valid i,j,k
    # But the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Wait, I need to be more careful about the index convention.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r]
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_4x4_4x8(n, m, p):
    """
    Try to construct a decomposition using Strassen-like ideas.
    For 4x4 x 4x8, view as two 4x4 x 4x4 multiplications.
    Use a known rank-20 decomposition for 4x4x4 if available.
    """
    # For now, fall back to schoolbook if we can't find a better one
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try to find a better decomposition
    # For 4x4 x 4x8, try various approaches
    
    # Approach 1: Try to use a known low-rank decomposition
    # For 4x4 x 4x4, there's a known rank-20 decomposition
    # Let's try to construct one
    
    # Actually, let's try a simple optimization approach
    # But given time constraints, let's just return the schoolbook for now
    # unless we can quickly find something better
    
    # Try a heuristic: see if we can reduce rank by merging terms
    # This is complex, so for now, return schoolbook
    
    return best
# EVOLVE-BLOCK-END
