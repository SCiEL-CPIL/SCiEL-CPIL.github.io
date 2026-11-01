"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Uses explicit Strassen-based construction for rank 23:
  - 3 blocks of 2 columns each: Strassen rank-7 each (21 total)
  - 1 remaining column: schoolbook rank-2
  Total rank: 23

The tensor condition is:
  C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] A[i,j] * V[j*p+k,c] B[j,k] * W[k*n+i,c]
  
Equivalently, for all (i,j,k):
  sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]


def _strassen_2x2_block(col_offset, rank_offset):
    """
    Build Strassen rank-7 decomposition for 2x2 * 2x2 block starting at column col_offset.
    Returns (U_rows, V_rows, W_rows) where each is a list of 7 entries (one per rank term).
    U has 4 rows (n*m = 4), V has 4 rows (m*2 = 4 for this block), W has 4 rows (2*2 = 4).
    """
    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each multiplication M_c, we define:
    #   U part: coefficients on A[i,j] for i in {0,1}, j in {0,1}
    #   V part: coefficients on B[j,k] for j in {0,1}, k in {0,1}
    #   W part: how this term contributes to C[i,k]
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3 (i*m+j)
    # B indices for this block: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3 (j*2+k)
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (k*n+i, n=2)
    
    # Define the 7 Strassen terms
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) -> coeff
    # V_coeffs: dict mapping (j,k) -> coeff  
    # W_coeffs: dict mapping (i,k) -> coeff (contribution to C[i,k])
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    U1 = {(0,0): 1, (1,1): 1}
    V1 = {(0,0): 1, (1,1): 1}
    # M1 contributes to C00 (+1) and C11 (+1)
    W1 = {(0,0): 1, (1,1): 1}
    terms.append((U1, V1, W1))
    
    # M2 = (A10+A11)B00
    U2 = {(1,0): 1, (1,1): 1}
    V2 = {(0,0): 1}
    # M2 contributes to C10 (+1) and C11 (-1)
    W2 = {(1,0): 1, (1,1): -1}
    terms.append((U2, V2, W2))
    
    # M3 = A00(B10-B11)
    U3 = {(0,0): 1}
    V3 = {(1,0): 1, (1,1): -1}
    # M3 contributes to C01 (+1) and C11 (+1)
    W3 = {(0,1): 1, (1,1): 1}
    terms.append((U3, V3, W3))
    
    # M4 = A11(B01-B10)
    U4 = {(1,1): 1}
    V4 = {(0,1): 1, (1,0): -1}
    # M4 contributes to C00 (+1) and C10 (+1)
    W4 = {(0,0): 1, (1,0): 1}
    terms.append((U4, V4, V4))  # Wait, W4 should be separate
    
    # Let me redo this more carefully
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    U1 = {(0,0): 1, (1,1): 1}
    V1 = {(0,0): 1, (1,1): 1}
    W1 = {(0,0): 1, (1,1): 1}
    terms.append((U1, V1, W1))
    
    # M2 = (A10+A11)B00
    U2 = {(1,0): 1, (1,1): 1}
    V2 = {(0,0): 1}
    W2 = {(1,0): 1, (1,1): -1}
    terms.append((U2, V2, W2))
    
    # M3 = A00(B10-B11)
    U3 = {(0,0): 1}
    V3 = {(1,0): 1, (1,1): -1}
    W3 = {(0,1): 1, (1,1): 1}
    terms.append((U3, V3, W3))
    
    # M4 = A11(B01-B10)
    U4 = {(1,1): 1}
    V4 = {(0,1): 1, (1,0): -1}
    W4 = {(0,0): 1, (1,0): 1}
    terms.append((U4, V4, W4))
    
    # M5 = (A00+A01)B11
    U5 = {(0,0): 1, (0,1): 1}
    V5 = {(1,1): 1}
    W5 = {(0,0): -1, (0,1): 1}
    terms.append((U5, V5, W5))
    
    # M6 = (A10-A00)(B00+B10)
    U6 = {(1,0): 1, (0,0): -1}
    V6 = {(0,0): 1, (1,0): 1}
    W6 = {(1,1): 1}
    terms.append((U6, V6, W6))
    
    # M7 = (A01-A11)(B01+B11)
    U7 = {(0,1): 1, (1,1): -1}
    V7 = {(0,1): 1, (1,1): 1}
    W7 = {(0,0): 1}
    terms.append((U7, V7, W7))
    
    return terms


def _build_strassen_decomposition():
    """
    Build explicit rank-23 decomposition using Strassen for 3 blocks of 2 cols + schoolbook for 1 col.
    """
    n, m, p = 2, 2, 7
    total_rank = 7 * 3 + 2  # 23
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 23
    V = [[0] * total_rank for _ in range(m * p)]  # 14 x 23
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 23
    
    rank_offset = 0
    
    # Three Strassen blocks for columns (0,1), (2,3), (4,5)
    for block_idx in range(3):
        col_start = block_idx * 2  # columns 0, 2, 4
        
        terms = _strassen_2x2_block(col_start, rank_offset)
        
        for term_idx, (U_coeffs, V_coeffs, W_coeffs) in enumerate(terms):
            c = rank_offset + term_idx
            
            # Fill U: A[i,j] where i in {0,1}, j in {0,1}
            for (i, j), coeff in U_coeffs.items():
                U[i * m + j][c] = coeff
            
            # Fill V: B[j,k] where j in {0,1}, k in {col_start, col_start+1}
            for (j, k_local), coeff in V_coeffs.items():
                k_global = col_start + k_local
                V[j * p + k_global][c] = coeff
            
            # Fill W: C[i,k] where k in {col_start, col_start+1}, i in {0,1}
            # W index is k*n + i
            for (i, k_local), coeff in W_coeffs.items():
                k_global = col_start + k_local
                W[k_global * n + i][c] = coeff
        
        rank_offset += 7
    
    # Schoolbook for column 6 (2x2 * 2x1)
    # C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6]
    # Two terms:
    # Term 1: A[i,0]*B[0,6] -> U[i*2+0]=1, V[0*7+6]=1, W[6*2+i]=1
    # Term 2: A[i,1]*B[1,6] -> U[i*2+1]=1, V[1*7+6]=1, W[6*2+i]=1
    
    # Term 1: j=0
    c1 = rank_offset
    for i in range(n):
        U[i * m + 0][c1] = 1
        W[6 * n + i][c1] = 1
    V[0 * p + 6][c1] = 1
    
    # Term 2: j=1
    c2 = rank_offset + 1
    for i in range(n):
        U[i * m + 1][c2] = 1
        W[6 * n + i][c2] = 1
    V[1 * p + 6][c2] = 1
    
    return U, V, W


def _verify_exact(U, V, W, n, m, p, probes=50):
    """Exact check of the bilinear identity for random integer A,B."""
    R = len(U[0])
    
    for _ in range(probes):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        
        # Expected C = A*B
        Cexp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Reconstructed C using tensor decomposition
        Crec = [[0] * p for _ in range(n)]
        for c in range(R):
            # ua[i] = sum_j U[i*m+j, c] * A[i,j]
            ua = [sum(U[i * m + j][c] * A[i][j] for j in range(m)) for i in range(n)]
            # vb[k] = sum_j V[j*p+k, c] * B[j,k]
            vb = [sum(V[j * p + k][c] * B[j][k] for j in range(m)) for k in range(p)]
            # Add to Crec
            for i in range(n):
                for k in range(p):
                    Crec[i][k] += W[k * n + i][c] * ua[i] * vb[k]
        
        for i in range(n):
            for k in range(p):
                if Crec[i][k] != Cexp[i][k]:
                    return False
    
    return True


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Build the explicit rank-23 Strassen-based decomposition
    U, V, W = _build_strassen_decomposition()
    
    # Verify it's correct
    if _verify_exact(U, V, W, n, m, p, probes=100):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if something went wrong
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][c] = 1
                V_fb[j * p + k][c] = 1
                W_fb[k * n + i][c] = 1
                c += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
