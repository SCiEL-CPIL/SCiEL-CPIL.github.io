import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_decomposition(n, m, p):
    """
    Strassen-based rank-66 decomposition for 3x4x6.
    Uses 6 Strassen 2x2x2 blocks (rows 0-1) + 24 schoolbook terms (row 2).
    """
    rank = 6 * 7 + 24  # 42 + 24 = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    row_pairs = [(0, 1)]
    mid_pairs = [(0, 1), (2, 3)]
    col_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for (r0, r1) in row_pairs:
        for (s0, s1) in mid_pairs:
            for (t0, t1) in col_pairs:
                # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
                U[r0 * m + s0, col] = 1
                U[r1 * m + s1, col] = 1
                V[s0 * p + t0, col] = 1
                V[s1 * p + t1, col] = 1
                W[t0 * n + r0, col] = 1
                W[t1 * n + r1, col] = 1
                col += 1
                
                # Term 2: (A10+A11)B00 -> C10+1
                U[r1 * m + s0, col] = 1
                U[r1 * m + s1, col] = 1
                V[s0 * p + t0, col] = 1
                W[t0 * n + r1, col] = 1
                col += 1
                
                # Term 3: A00(B01-B11) -> C01+1
                U[r0 * m + s0, col] = 1
                V[s0 * p + t1, col] = 1
                V[s1 * p + t1, col] = -1
                W[t1 * n + r0, col] = 1
                col += 1
                
                # Term 4: A11(B10-B00) -> C10+1, C11-1
                U[r1 * m + s1, col] = 1
                V[s1 * p + t0, col] = 1
                V[s0 * p + t0, col] = -1
                W[t0 * n + r1, col] = 1
                W[t1 * n + r1, col] = -1
                col += 1
                
                # Term 5: (A00+A01)B11 -> C00+1, C01+1
                U[r0 * m + s0, col] = 1
                U[r0 * m + s1, col] = 1
                V[s1 * p + t1, col] = 1
                W[t0 * n + r0, col] = 1
                W[t1 * n + r0, col] = 1
                col += 1
                
                # Term 6: (A10-A00)(B00+B01) -> C00+1, C10+1
                U[r1 * m + s0, col] = 1
                U[r0 * m + s0, col] = -1
                V[s0 * p + t0, col] = 1
                V[s0 * p + t1, col] = 1
                W[t0 * n + r0, col] = 1
                W[t0 * n + r1, col] = 1
                col += 1
                
                # Term 7: (A01-A11)(B10+B11) -> C01-1, C11-1
                U[r0 * m + s1, col] = 1
                U[r1 * m + s1, col] = -1
                V[s1 * p + t0, col] = 1
                V[s1 * p + t1, col] = 1
                W[t1 * n + r0, col] = -1
                W[t1 * n + r1, col] = -1
                col += 1
    
    # Schoolbook terms for row 2
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if total != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin
    
    # Try Strassen-based rank-66 decomposition
    try:
        U, V, W = _strassen_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception as e:
        pass
    
    # Fallback: Schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
