"""Block-decomposition program for 4x4 @ 4x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is (U dict, V dict, W dict) with integer entries."""
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M2 = (A21+A22)B11
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # M3 = A11(B21-B22)
    terms.append(({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # M4 = A22(B12-B11)
    terms.append(({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # M5 = (A11+A12)B22
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # M6 = (A21-A11)(B11+B12)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # M7 = (A12-A22)(B21+B22)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _build_4x4x4_rank20():
    """Build a rank-20 decomposition for 4x4x4 matrix multiplication.
    Uses 4 copies of Strassen 2x2x2 with shared terms.
    Returns list of 20 (U dict, V dict, W dict) terms."""
    S22 = _strassen_2x2()
    terms = []
    
    # Top-left 2x2x2 block: rows 0-1, cols 0-1
    for (u, v, w) in S22:
        terms.append((u, v, w))
    
    # Top-right 2x2x2 block: rows 0-1, cols 2-3
    for (u, v, w) in S22:
        u_new = {(i+0, j+2): val for (i, j), val in u.items()}
        v_new = {(i+2, j): val for (i, j), val in v.items()}
        w_new = {(i+2, j): val for (i, j), val in w.items()}
        terms.append((u_new, v_new, w_new))
    
    # Bottom-left 2x2x2 block: rows 2-3, cols 0-1
    for (u, v, w) in S22:
        u_new = {(i+2, j): val for (i, j), val in u.items()}
        v_new = {(i, j+2): val for (i, j), val in v.items()}
        w_new = {(i, j+2): val for (i, j), val in w.items()}
        terms.append((u_new, v_new, w_new))
    
    # Bottom-right 2x2x2 block: rows 2-3, cols 2-3
    for (u, v, w) in S22:
        u_new = {(i+2, j+2): val for (i, j), val in u.items()}
        v_new = {(i+2, j+2): val for (i, j), val in v.items()}
        w_new = {(i+2, j+2): val for (i, j), val in w.items()}
        terms.append((u_new, v_new, w_new))
    
    # This gives 28 terms, not 20. We need to share terms.
    # For now, return the 28-term decomposition. We'll optimize later.
    return terms

def _build_4x4x5_decomp():
    """Build a decomposition for 4x4 @ 4x5.
    Uses 4x4x4 decomposition (rank 28 for now) + 4 terms for the 5th column.
    Total rank = 28 + 4 = 32."""
    n, m, p = DIMENSIONS
    terms_4x4x4 = _build_4x4x4_rank20()
    terms = list(terms_4x4x4)
    
    # Add 4 terms for the 5th column: A(4x4) @ B(4x1) -> C(4x1)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        u = {(i, j): 1 for j in range(4)}
        v = {(j, 4): 1 for j in range(4)}
        w = {(4, i): 1}
        terms.append((u, v, w))
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of (U dict, V dict, W dict) terms to U, V, W matrices."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] += val
        for (j, k), val in v.items():
            V[j * p + k][r] += val
        for (k, i), val in w.items():
            W[k * n + i][r] += val
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j] * Va[j*p+k], Wa[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_halfint(U, V, W, tol=1e-6):
    for M in (U, V, W):
        for row in M:
            for x in row:
                if abs(2.0 * x - round(2.0 * x)) > tol:
                    return False
    return True

def _schoolbook():
    """Fallback schoolbook decomposition rank n*m*p = 80."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start = time.time()
    n, m, p = DIMENSIONS

    # Primary: 4x4x4 + 5th column decomposition (rank 32 for now)
    try:
        terms = _build_4x4x5_decomp()
        U, V, W = _terms_to_matrices(terms, n, m, p)
        if _verify(U, V, W, n, m, p) and _verify_halfint(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: schoolbook (rank 80)
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
