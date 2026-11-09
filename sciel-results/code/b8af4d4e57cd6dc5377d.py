import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for each (i, k), sum_j T[i,j,k] * A[i,j] * B[j,k] should give C[i,k]
    # Actually, the tensor should satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    # because each product A[i,j]*B[j,k] is a separate term in the sum
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    
    return max_err < 1e-10

def _build_block_decomposition(n, m, p):
    """
    Build a block-structured decomposition for 3x4x6 matrix multiplication.
    
    Split A (3x4) into A1 (2x4) and A2 (1x4)
    Split B (4x6) into B1 (4x3) and B2 (4x3)
    
    Subproblems:
    - C11 = A1*B1: 2x4 * 4x3 (use 12-term algorithm)
    - C12 = A1*B2: 2x4 * 4x3 (use 12-term algorithm)
    - C21 = A2*B1: 1x4 * 4x3 (use 6-term algorithm)
    - C22 = A2*B2: 1x4 * 4x3 (use 6-term algorithm)
    """
    # Start with schoolbook and try to find a lower rank decomposition
    # For now, return schoolbook as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication 3x4 by 4x6.
    Currently returns the schoolbook decomposition of rank 72.
    """
    n, m, p = DIMENSIONS
    result = _schoolbook_decomposition(n, m, p)
    return result

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
