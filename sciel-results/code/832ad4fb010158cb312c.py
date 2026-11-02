"""
Matrix multiplication 3x5 * 5x7 tensor decomposition.

Strategy: Direct sum decomposition by partitioning the 5-dimensional inner product
into two independent subspaces of dimensions 2 and 3.
- Block 1: 3x2 x 2x7 (inner dim 2). Rank = 3*2*7 = 42 (schoolbook).
- Block 2: 3x3 x 3x7 (inner dim 3). Rank = 3*3*7 = 63 (schoolbook).
Total rank = 42 + 63 = 105.

This is structurally identical to the schoolbook decomposition but explicitly
constructed as a direct sum. While the rank is the same (105), this approach
sets up the framework for potential improvements if better algorithms for the
sub-blocks are discovered. For now, it guarantees correctness and clarity.

We also include a verification step to ensure the decomposition is valid.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    
    Condition: for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partition the inner dimension m=5 into m1=2 and m2=3
    m1, m2 = 2, 3
    assert m1 + m2 == m
    
    # Block 1: 3 x m1 x 7 (A1: 3xm1, B1: m1xp)
    n1, m1_, p1 = n, m1, p
    rank1 = n1 * m1_ * p1
    
    # Block 2: 3 x m2 x 7 (A2: 3xm2, B2: m2xp)
    n2, m2_, p2 = n, m2, p
    rank2 = n2 * m2_ * p2
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Block 1: indices i in [0,n), j in [0,m1), k in [0,p) ---
    col = 0
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # --- Block 2: indices i in [0,n), j in [m1, m1+m2), k in [0,p) ---
    # j_global = m1 + j_local, where j_local in [0, m2)
    for i in range(n):
        for j_local in range(m2):
            j_global = m1 + j_local
            for k in range(p):
                # U index: i*m + j_global
                # V index: j_global*p + k
                # W index: k*n + i
                U[i * m + j_global][col] = 1
                V[j_global * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness (only if we have time)
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to pure schoolbook if verification fails (should not happen)
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
            result = {"U": U, "V": V, "W": W}
    
    return result
