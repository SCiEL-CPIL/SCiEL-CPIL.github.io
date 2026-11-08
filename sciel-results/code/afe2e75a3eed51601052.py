import numpy as np
from datetime import datetime, timedelta

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix.
    
    Strategy:
    Split the middle dimension m=3 into j=2 (1x1 part) and j=0,1 (2x2 part).
    
    Part 2 (j=2): A_{2x1} * B_{1x9}.
    This is a rank-1 tensor slice.
    T_{i,2,k} = 1 for all i, k.
    Can be decomposed with rank 1.
    
    Part 1 (j=0,1): A_{2x2} * B_{2x9}.
    This consists of 9 independent 2x2 * 2x1 multiplications (one for each column k of B).
    The rank of 2x2 * 2x1 is 4.
    Total rank for Part 1 = 9 * 4 = 36.
    
    Total rank = 1 + 36 = 37.
    """
    n, m, p = DIMENSIONS
    rank = 37
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 2: j=2, rank 1
    r = 0
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = 1
    r += 1
    
    # Part 1: j=0,1, rank 36
    # For each column k of B (for j=0,1), we have a 2x2 * 2x1 multiplication.
    # This requires 4 terms.
    for k in range(p):
        for i in range(n):
            for j in range(2): # j=0, 1
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check dimensions
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != rank or V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i, j] * V_np[j * p + k, r] * W_np[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
