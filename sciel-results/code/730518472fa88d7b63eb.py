"""Rewritten program for matrix multiplication 4x4 * 4x7 tensor decomposition search."""
import time
import random
import itertools
import math
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    # Try Strassen-based decomposition for 4x4 * 4x4 part
    # and schoolbook for 4x4 * 4x3 part
    U, V, W = _strassen_4x4_4x7(n, m, p)

    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x2(n, m, p):
    """Strassen decomposition for 2x2 * 2x2 -> rank 7"""
    # Standard Strassen indices
    # U rows: (i,j) -> 4 entries
    # V rows: (j,k) -> 4 entries  
    # W rows: (k,n+i) -> 4 entries
    # 7 Strassen products
    strassen_terms = [
        # (U_mask, V_mask, W_mask)
        # M1 = (A00+A11)(B00+B11)
        ([0, 3], [0, 3], [0, 1, 2, 3]),
        # M2 = (A10+A11)B00
        ([2, 3], [0, 2], [1, 3]),
        # M3 = A00(B10-B11)
        ([0], [1, 3], [0, 2]),
        # M4 = A11(B01-B10)
        ([3], [0, 1], [1, 3]),
        # M5 = (A00+A01)B11
        ([0, 1], [1, 3], [2, 3]),
        # M6 = (A10-A00)(B00+B01)
        ([0, 2], [0, 1], [1, 2]),
        # M7 = (A01-A10)(B10+B11)
        ([1, 2], [1, 3], [0, 3]),
    ]
    return strassen_terms

def _build_4x4_4x4():
    """Build 4x4 * 4x4 decomposition using Strassen blocks."""
    # Split 4x4 into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # 
    # Use Strassen for each 2x2*2x2 product
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Actually, for 4x4*4x4, we can use a recursive Strassen approach
    # or just use the 16 schoolbook terms per 2x2 block = 16*4 = 64
    # But Strassen gives 7 per 2x2*2x2, and we need 4 such products = 28
    # Wait, that's not right. Let me think more carefully.
    #
    # For 4x4 * 4x4, we can partition into 2x2 blocks:
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2*2x2 product has rank 7 (Strassen)
    # But we have 8 such products (4 output blocks, each needs 2 products)
    # So total rank = 8 * 7 = 56
    #
    # Alternatively, we can use a direct Strassen on the 4x4 level
    # which would give rank 7^2 = 49 if we can do it properly
    # But 4x4 isn't a power of 2 in the right way for direct Strassen
    #
    # Let's use the 56-rank approach for now
    
    rank = 56
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16
    V = [[0] * rank for _ in range(16)]  # 4*4 = 16
    W = [[0] * rank for _ in range(16)]  # 4*4 = 16
    
    col = 0
    
    # Block structure:
    # A block (bi, bj) where bi, bj in {0,1} -> 2x2 block
    # Within block, local indices (li, lj) in {0,1}
    # Global A index: (bi*2+li) * 4 + (bj*2+lj) = bi*8 + bj*2 + li*4 + lj... 
    # Wait, A is 4x4, indexed as A[i,j] where i,j in 0..3
    # Block A[bi, bj] is 2x2 with local indices (li, lj)
    # Global index: i = bi*2 + li, j = bj*2 + lj
    # A row index in U: i * m + j = (bi*2+li) * 4 + (bj*2+lj)
    
    # For each output block (ci, cj) and each of the 2 Strassen products:
    # C[ci, cj] = A[ci, 0]*B[0, cj] + A[ci, 1]*B[1, cj]
    
    strassen_2x2 = _strassen_2x2_2x2(2, 2, 2)
    
    for ci in range(2):
        for cj in range(2):
            for prod_idx in range(2):
                # prod_idx 0: A[ci, 0] * B[0, cj]
                # prod_idx 1: A[ci, 1] * B[1, cj]
                ai = ci
                bj = prod_idx  # 0 or 1
                bi = prod_idx  # 0 or 1 for B
                # Wait, I need to be more careful
                # C[ci, cj] = sum_{mid} A[ci, mid] * B[mid, cj]
                # where mid is block index 0 or 1
                # So for prod_idx=0: A[ci, 0] * B[0, cj]
                # For prod_idx=1: A[ci, 1] * B[1, cj]
                
                for s in range(7):
                    u_mask, v_mask, w_mask = strassen_2x2[s]
                    
                    # A block: rows ci*2+li, cols bj*2+lj for li,lj in 0,1
                    # B block: rows bi*2+li, cols cj*2+lj for li,lj in 0,1
                    # C block: rows ci*2+li, cols cj*2+lj for li,lj in 0,1
                    
                    for li in range(2):
                        for lj in range(2):
                            # A global index
                            a_i = ci * 2 + li
                            a_j = bj * 2 + lj
                            a_row = a_i * 4 + a_j  # i * m + j where m=4
                            
                            # B global index
                            b_i = bi * 2 + li
                            b_j = cj * 2 + lj
                            b_row = b_i * 4 + b_j  # j * p + k where p=4
                            
                            # C global index
                            c_i = ci * 2 + li
                            c_j = cj * 2 + lj
                            c_row = c_j * 4 + c_i  # k * n + i where n=4
                            
                            if li in u_mask and lj in u_mask:
                                U[a_row][col] = 1
                            if li in v_mask and lj in v_mask:
                                V[b_row][col] = 1
                            if li in w_mask and lj in w_mask:
                                W[c_row][col] = 1
                    
                    col += 1
    
    return U, V, W, rank

def _schoolbook_4x4_4x3():
    """Schoolbook for 4x4 * 4x3."""
    rank = 4 * 4 * 3  # 48
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(12)]  # 4*3
    W = [[0] * rank for _ in range(12)]  # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return U, V, W, rank

def _strassen_4x4_4x7(n, m, p):
    """
    Build 4x4 * 4x7 decomposition.
    Split B into B_left (4x4) and B_right (4x3).
    Use Strassen for 4x4 * 4x4 (rank 56) and schoolbook for 4x4 * 4x3 (rank 48).
    Total rank: 104
    """
    # 4x4 * 4x4 part
    U1, V1, W1, rank1 = _build_4x4_4x4()
    
    # 4x4 * 4x3 part
    U2, V2, W2, rank2 = _schoolbook_4x4_4x3()
    
    total_rank = rank1 + rank2  # 56 + 48 = 104
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]  # 4*7
    W = [[0] * total_rank for _ in range(28)]  # 7*4
    
    # Copy 4x4*4x4 part
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    
    for i in range(16):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    
    for i in range(16):
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    # Copy 4x4*4x3 part (columns 4-6 of B)
    offset = rank1
    for i in range(16):
        for c in range(rank2):
            U[i][offset + c] = U2[i][c]
    
    for i in range(12):
        for c in range(rank2):
            V[i + 12][offset + c] = V2[i][c]  # B rows 12-23 correspond to j=0..3, k=4..6
    
    for i in range(12):
        for c in range(rank2):
            W[i + 16][offset + c] = W2[i][c]  # W rows 16-27 correspond to k=4..6, i=0..3
    
    return U, V, W

def _schoolbook(n, m, p):
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

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
