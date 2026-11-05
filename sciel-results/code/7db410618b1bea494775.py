"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-63 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: partition 6x6 into 2x2 blocks of 3x3 matrices, use schoolbook
    for the 3x3 block structure (9 block products) and rank-7 Strassen for
    each 3x3 block multiplication.
    
    Total rank = 9 * 7 = 63.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 63
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 63)
    V = [[0] * rank for _ in range(m * p)]  # (36, 63)
    W = [[0] * rank for _ in range(p * n)]  # (36, 63)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For a 2x2 block at position (bi, bj) in the 2x2 block grid:
    # Block rows: 3*bi, 3*bi+1, 3*bi+2
    # Block cols: 3*bj, 3*bj+1, 3*bj+2
    # A11 = A[3*bi:3*bi+3, 3*bj:3*bj+3], etc.
    
    # Strassen factors for one 2x2 multiplication:
    # Each term t has U_factors (list of (row, coeff)) and V_factors (list of (col, coeff))
    # and contributes to W with W_factors (list of (row, col, coeff))
    
    # Term definitions: (U_row_offsets, U_coeffs, V_col_offsets, V_coeffs)
    # where offsets are relative to block top-left, 0=A11, 1=A12, 2=A21, 3=A22
    # For U: rows are (row, col) in 2x2 block -> global row = 3*bi + row, global col = 3*bj + col
    # For V: cols are (row, col) in 2x2 block -> global row = 3*bk + row, global col = 3*bj + col
    # V index is j*p+k where j is column of A, k is row of B.
    # So for block (bi, bj) in A and (bk, bj) in B:
    # A element at block position (r, c) -> global (3*bi+r, 3*bj+c) -> U index (3*bi+r)*6 + (3*bj+c)
    # B element at block position (r, c) -> global (3*bk+r, 3*bj+c) -> V index (3*bj+c)*6 + (3*bk+r)
    # C element at block position (r, c) -> global (3*bi+r, 3*bj+c) -> W index (3*bj+c)*6 + (3*bi+r)
    
    # Strassen terms: each term is a product of a linear combination of A elements
    # and a linear combination of B elements.
    
    # Let's define the 7 Strassen terms as (U_comb, V_comb) where each is a list of
    # (offset, coeff) for the 2x2 block elements.
    
    # A offsets: 0=(0,0)=A11, 1=(0,1)=A12, 2=(1,0)=A21, 3=(1,1)=A22
    # B offsets: 0=(0,0)=B11, 1=(0,1)=B12, 2=(1,0)=B21, 3=(1,1)=B22
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        # M2 = (A21+A22)B11
        (([2, 3], [1, 1]), ([0], [1])),
        # M3 = A11(B12-B22)
        (([0], [1]), ([1, 3], [1, -1])),
        # M4 = A22(B21-B11)
        (([3], [1]), ([2, 0], [1, -1])),
        # M5 = (A11+A12)B22
        (([0, 1], [1, 1]), ([3], [1])),
        # M6 = (A21-A11)(B11+B12)
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        # M7 = (A12-A22)(B21+B22)
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions for each Strassen term:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # C offsets: 0=(0,0)=C11, 1=(0,1)=C12, 2=(1,0)=C21, 3=(1,1)=C22
    
    c_contributions = [
        # For each Strassen term t, list of (c_offset, coeff)
        [(0, 1), (3, 1)],   # M1 -> C11, C22
        [(2, 1), (3, -1)],  # M2 -> C21, C22
        [(1, 1), (3, 1)],   # M3 -> C12, C22
        [(0, 1), (2, 1)],   # M4 -> C11, C21
        [(0, -1), (1, 1)],  # M5 -> C11, C12
        [(3, 1)],           # M6 -> C22
        [(0, 1)],           # M7 -> C11
    ]
    
    col = 0
    for bi in range(2):  # block row of A
        for bk in range(2):  # block col of A = block row of B
            for bj in range(2):  # block col of B
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Set U factors
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 3 * bi + 3 * r
                        global_col = 3 * bk + 3 * c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    # Set V factors
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 3 * bk + 3 * r
                        global_col = 3 * bj + 3 * c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    # Set W factors
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 3 * bj + 3 * c
                        global_col = 3 * bi + 3 * r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
