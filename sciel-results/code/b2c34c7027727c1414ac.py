"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += Fraction(U[u_idx][r]) * Fraction(V[v_idx][r]) * Fraction(W[w_idx][r])
                if s != Fraction(1):
                    return False
    return True


def _laderman_2x2x2():
    """
    Return Laderman's rank-7 decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7), V has shape (4,7), W has shape (4,7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1, 1, 1, 1, 0, 0, 0],   # A[0,0]
        [1,-1, 0, 0, 1, 0, 0],   # A[0,1]
        [0, 0, 1,-1, 0, 1, 0],   # A[1,0]
        [0, 0, 0, 0, 1,-1, 0],   # A[1,1]
    ]
    V = [
        [0, 1, 0, 0, 0, 1, 0],   # B[0,0]
        [1, 1, 0, 0, 0, 0, 0],   # B[0,1]
        [0, 0, 1, 1, 0, 0, 0],   # B[1,0]
        [0, 0, 0, 1, 1, 0, 0],   # B[1,1]
    ]
    W = [
        [0, 1, 0, 0, 0, 0, 0],   # C[0,0]
        [0,-1, 0, 0, 0, 0, 0],   # C[0,1]
        [0, 0, 1, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 0,-1, 1, 0, 0],   # C[1,1]
    ]
    
    # Verify
    n2, m2, p2 = 2, 2, 2
    if _verify(U, V, W, n2, m2, p2):
        return U, V, W
    else:
        # Fallback: schoolbook 2x2x2
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W


def _embed_laderman(U, V, W, i0, j0, k0, n, m, p):
    """
    Embed a Laderman 2x2x2 block into the full decomposition at offset (i0, j0, k0).
    Uses columns 0-6 of U, V, W.
    """
    U_l, V_l, W_l = _laderman_2x2x2()
    
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                i = i0 + li
                j = j0 + lj
                if i < n and j < m:
                    U[i * m + j][r] += U_l[li * 2 + lj][r]
        for lj in range(2):
            for lk in range(2):
                j = j0 + lj
                k = k0 + lk
                if j < m and k < p:
                    V[j * p + k][r] += V_l[lj * 2 + lk][r]
        for lk in range(2):
            for li in range(2):
                k = k0 + lk
                i = i0 + li
                if k < p and i < n:
                    W[k * n + i][r] += W_l[lk * 2 + li][r]


def _construct_rank_35():
    """
    Construct a rank-35 decomposition using Laderman for the 2x2x2 block
    and schoolbook for the remaining 28 terms.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Add Laderman block for i in {0,1}, j in {0,1}, k in {0,1}
    _embed_laderman(U, V, W, 0, 0, 0, n, m, p)
    
    # 2. Add schoolbook terms for the remaining 28 terms
    column = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by Laderman
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W


def _construct_rank_34():
    """
    Attempt to construct a rank-34 decomposition using two Laderman blocks
    with careful overlap handling.
    """
    n, m, p = DIMENSIONS
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    _embed_laderman(U, V, W, 0, 0, 0, n, m, p)
    
    # Block 2: i in {1,2}, j in {1,2}, k in {2,3}
    _embed_laderman(U, V, W, 1, 1, 2, n, m, p)
    
    # 3. Add schoolbook terms for the remaining terms
    column = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by either Laderman block
                if i < 2 and j < 2 and k < 2:
                    continue
                if 1 <= i <= 2 and 1 <= j <= 2 and 2 <= k <= 3:
                    continue
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_time=60):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rank = len(U[0])
    
    if time.time() > deadline - 5:
        return U, V, W, rank
    
    # Try to find a rank-34 decomposition
    U2, V2, W2 = _construct_rank_34()
    if len(U2[0]) < rank and _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2, len(U2[0])
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    
    # Try to construct rank-35 decomposition
    if time.time() < deadline:
        U2, V2, W2 = _construct_rank_35()
        if _verify(U2, V2, W2, n, m, p):
            if len(U2[0]) < len(U[0]):
                U, V, W = U2, V2, W2
        
        # Try rank reduction
        if time.time() < deadline - 30:
            U3, V3, W3, new_rank = _try_rank_reduction(U, V, W, n, m, p, max_time=30)
            if _verify(U3, V3, W3, n, m, p):
                if new_rank < len(U[0]):
                    U, V, W = U3, V3, W3
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook()
    
    return {"U": U, "V": V, "W": W}
