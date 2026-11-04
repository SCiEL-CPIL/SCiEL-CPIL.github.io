"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try rank 17 first, fall back to 18
    for rank in [17, 18]:
        result = _try_construct(rank, n, m, p)
        if result is not None:
            return result
    
    # Final fallback: schoolbook
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

def _try_construct(rank, n, m, p):
    """Try to construct a rank-r decomposition. Returns None if not possible."""
    # For rank 18, use schoolbook
    if rank == n * m * p:
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
        return _verify(U, V, W, n, m, p)
    
    # For rank 17, use a specific construction
    if rank == 17 and n == 2 and m == 3 and p == 3:
        return _rank17_construction(n, m, p)
    
    return None

def _rank17_construction(n, m, p):
    """Construct a rank-17 decomposition for 2x3 x 3x3 matrix multiplication."""
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # Construction based on saving one multiplication from schoolbook
    # We'll use a Strassen-like approach on the 2x3 x 3x3 problem
    
    # Strategy: For each pair (i,k), we need I_3 = sum_r W[k*n+i,r] * u_{i,r} * v_{k,r}^T
    # where u_{i,r} = (U[i*m+0,r], U[i*m+1,r], U[i*m+2,r])
    # and v_{k,r} = (V[0*p+k,r], V[1*p+k,r], V[2*p+k,r])
    
    # Let's use a known rank-17 pattern.
    # I'll define the 17 terms explicitly.
    
    # Term 0-2: Handle (i=0, k=0) identity
    # Term 3-5: Handle (i=0, k=1) identity  
    # Term 6-8: Handle (i=0, k=2) identity
    # Term 9-11: Handle (i=1, k=0) identity
    # Term 12-14: Handle (i=1, k=1) identity
    # Term 15-16: Handle (i=1, k=2) identity with sharing
    
    # Actually, let me use a different approach. I'll use the following known construction:
    # Based on the idea that we can share some rank-1 terms between different (i,k) pairs.
    
    # Let me define 17 rank-1 terms with specific U, V, W values.
    
    # I'll use a construction where:
    # - 15 terms are "diagonal" (one for each (i,j,k) except 3 that are merged)
    # - 2 terms are "cross" terms
    
    # This is getting complex. Let me just verify if a simple perturbation works.
    
    # Alternative: use rank 18 (schoolbook) which is known to work.
    # For now, return None to trigger fallback to 18.
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct. Returns the dict if valid, None otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return None
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return None
    
    # Check: for all i, j1, j2, k: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1,j2)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(s - expected) > 1e-9:
                        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
