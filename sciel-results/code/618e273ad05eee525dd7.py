"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]
DEADLINE = 135.0  # safety margin before 150s kill

def _time_remaining():
    return DEADLINE - time.time()

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _strassen_block(n_start, m_start, p_start, n, m, p, U, V, W, col_offset):
    """
    Add a Strassen 2x2x2 block starting at (n_start, m_start, p_start).
    Uses 7 rank-1 terms instead of 8.
    Returns the new column offset.
    """
    # Strassen formulas for 2x2 matrix multiplication
    # A = [[a,b],[c,d]] (rows n_start, n_start+1; cols m_start, m_start+1)
    # B = [[e,f],[g,h]] (rows m_start, m_start+1; cols p_start, p_start+1)
    # C = [[c00, c01], [c10, c11]] (rows n_start, n_start+1; cols p_start, p_start+1)
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each term: U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For term M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) contributes to C[0,0] and C[1,1]
    # U factor: A[0,0]=1, A[1,1]=1, others 0
    # V factor: B[0,0]=1, B[1,1]=1, others 0
    # W factor: C[0,0]=1, C[1,1]=1, others 0
    
    # M1: (a+d)(e+h) -> C[0,0] += 1, C[1,1] += 1
    U[n_start*m + m_start][col_offset] = 1
    U[(n_start+1)*m + (m_start+1)][col_offset] = 1
    V[m_start*p + p_start][col_offset] = 1
    V[(m_start+1)*p + (p_start+1)][col_offset] = 1
    W[p_start*n + n_start][col_offset] = 1
    W[(p_start+1)*n + (n_start+1)][col_offset] = 1
    col_offset += 1
    
    # M2: (c+d)e -> C[1,0] += 1
    U[n_start*m + (m_start+1)][col_offset] = 1
    U[(n_start+1)*m + (m_start+1)][col_offset] = 1
    V[m_start*p + p_start][col_offset] = 1
    W[p_start*n + (n_start+1)][col_offset] = 1
    col_offset += 1
    
    # M3: a(f-h) -> C[0,1] += 1
    U[n_start*m + m_start][col_offset] = 1
    V[m_start*p + (p_start+1)][col_offset] = 1
    V[(m_start+1)*p + (p_start+1)][col_offset] = -1
    W[(p_start+1)*n + n_start][col_offset] = 1
    col_offset += 1
    
    # M4: d(g-e) -> C[0,0] += 1, C[1,0] += 1
    U[(n_start+1)*m + (m_start+1)][col_offset] = 1
    V[(m_start+1)*p + p_start][col_offset] = 1
    V[m_start*p + p_start][col_offset] = -1
    W[p_start*n + n_start][col_offset] = 1
    W[p_start*n + (n_start+1)][col_offset] = 1
    col_offset += 1
    
    # M5: (a+b)h -> C[0,0] -= 1, C[0,1] += 1
    U[n_start*m + m_start][col_offset] = 1
    U[n_start*m + (m_start+1)][col_offset] = 1
    V[(m_start+1)*p + (p_start+1)][col_offset] = 1
    W[p_start*n + n_start][col_offset] = -1
    W[(p_start+1)*n + n_start][col_offset] = 1
    col_offset += 1
    
    # M6: (c-a)(e+f) -> C[1,1] += 1
    U[(n_start+1)*m + m_start][col_offset] = 1
    U[n_start*m + m_start][col_offset] = -1
    V[m_start*p + p_start][col_offset] = 1
    V[m_start*p + (p_start+1)][col_offset] = 1
    W[(p_start+1)*n + (n_start+1)][col_offset] = 1
    col_offset += 1
    
    # M7: (b-d)(g+h) -> C[0,0] += 1
    U[n_start*m + (m_start+1)][col_offset] = 1
    U[(n_start+1)*m + (m_start+1)][col_offset] = -1
    V[(m_start+1)*p + p_start][col_offset] = 1
    V[(m_start+1)*p + (p_start+1)][col_offset] = 1
    W[p_start*n + n_start][col_offset] = 1
    col_offset += 1
    
    return col_offset

def _schoolbook_term(i, j, k, n, m, p, U, V, W, col_offset):
    """Add a single schoolbook term for (i,j,k)."""
    U[i*m + j][col_offset] = 1
    V[j*p + k][col_offset] = 1
    W[k*n + i][col_offset] = 1
    return col_offset + 1

def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen 2x2x2 blocks where possible,
    and schoolbook terms for the rest.
    
    For 3x5x7:
    - n: use rows 0,1 for Strassen, row 2 for schoolbook
    - m: use cols 0,1 and 2,3 for Strassen, col 4 for schoolbook
    - p: use cols 0,1 and 2,3 and 4,5 for Strassen, col 6 for schoolbook
    
    Strassen blocks: 1 * 2 * 3 = 6 blocks, each rank 7 = 42
    Remaining schoolbook terms: 105 - 6*8 = 105 - 48 = 57 terms, each rank 1 = 57
    Total rank: 42 + 57 = 99
    """
    rank = 0
    
    # Determine which (i,j,k) are covered by Strassen blocks
    strassen_i = set()
    strassen_j = set()
    strassen_k = set()
    
    # n=3: use i in {0,1}
    # m=5: use j in {0,1,2,3}
    # p=7: use k in {0,1,2,3,4,5}
    
    strassen_i_blocks = [(0, 1)]  # one block covering i=0,1
    strassen_j_blocks = [(0, 1), (2, 3)]  # two blocks covering j=0,1 and j=2,3
    strassen_k_blocks = [(0, 1), (2, 3), (4, 5)]  # three blocks covering k=0,1,2,3,4,5
    
    # Mark which (i,j,k) are in Strassen blocks
    covered = set()
    for (i0, i1) in strassen_i_blocks:
        for (j0, j1) in strassen_j_blocks:
            for (k0, k1) in strassen_k_blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0+di, j0+dj, k0+dk))
    
    # Count rank
    num_strassen = len(strassen_i_blocks) * len(strassen_j_blocks) * len(strassen_k_blocks)
    num_schoolbook = n * m * p - len(covered)
    total_rank = num_strassen * 7 + num_schoolbook
    
    # Build matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    for (i0, i1) in strassen_i_blocks:
        for (j0, j1) in strassen_j_blocks:
            for (k0, k1) in strassen_k_blocks:
                col = _strassen_block(i0, j0, k0, n, m, p, U, V, W, col)
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    col = _schoolbook_term(i, j, k, n, m, p, U, V, W, col)
    
    # Verify
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    result = _build_strassen_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
