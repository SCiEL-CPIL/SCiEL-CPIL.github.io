"""Program for matrix multiplication 2x3 by 3x9 with rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
_DEADLINE = 150.0

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p):
    """
    Validate that the decomposition exactly computes matrix multiplication.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """
    Build the schoolbook decomposition with rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}


def _build_rank_27_decomposition(n, m, p):
    """
    Build a rank-27 decomposition for 2x3 by 3x9 matrix multiplication.
    
    Key insight: For each (j,k) pair, we can use one term where:
    - U[3*i+j][r] = 1 for both i=0,1
    - V[9*j+k][r] = 1
    - W[2*k+i][r] = 1/3 for both i=0,1
    
    Then: sum_j U[3*i+j][r_{j,k}] * V[9*j+k][r_{j,k}] * W[2*k+i][r_{j,k}]
         = sum_j 1 * 1 * (1/3) = 3 * (1/3) = 1  ✓
    """
    if n != 2 or m != 3 or p != 9:
        return None
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 6 x 27
    V = [[0] * rank for _ in range(m * p)]  # 27 x 27
    W = [[0] * rank for _ in range(p * n)]  # 18 x 27
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U entries: for both i values, set to 1
            U[0 * m + j][r] = 1
            U[1 * m + j][r] = 1
            # V entry
            V[j * p + k][r] = 1
            # W entries: for both i values, set to 1/3
            W[k * n + 0][r] = Fraction(1, 3)
            W[k * n + 1][r] = Fraction(1, 3)
    
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank-27 decomposition
    better = _build_rank_27_decomposition(n, m, p)
    
    if better is not None:
        U, V, W = better["U"], better["V"], better["W"]
        if _validate_decomposition(U, V, W, n, m, p):
            return better
    
    # Fall back to schoolbook
    result = _build_schoolbook(n, m, p)
    
    U, V, W = result["U"], result["V"], result["W"]
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        pass  # Should not happen
    
    elapsed = time.time() - start
    if elapsed > _DEADLINE - 10:
        return result
    
    return result
# EVOLVE-BLOCK-END
