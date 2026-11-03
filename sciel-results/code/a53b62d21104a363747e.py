import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    # Components:
    # 1. M_4,4,4 (Rank 28 via 4x Strassen 2x2)
    # 2. M_4,4,1 (Rank 4)
    # 3. M_4,1,5 (Rank 4)
    
    rank_total = 36
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # 1. M_4,4,4: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # Use 4 Strassen 2x2 decompositions for the 4 quadrants of the 4x4 matrices?
    # No, 4x4 matrix multiplication can be done by 4 independent 2x2 multiplications.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # Each Cxy is 2x2.
    # Each product Aij Bkl is 2x2 * 2x2. Rank 7.
    # Total 8 products? No.
    # C11 uses A11 B11 and A12 B21.
    # This is not just 4 independent 2x2 mults.
    # Standard block multiplication:
    # C11 = A11 B11 + A12 B21
    # This requires 2 multiplications of 2x2 matrices.
    # Total 8 multiplications of 2x2 matrices.
    # Rank 8 * 7 = 56.
    # This is worse than schoolbook 64? No, 56 < 64.
    # But we can do better.
    # Strassen for 4x4 directly?
    # Or just use the 28 rank bound if available?
    # Actually, Rank(M_4,4,4) is 28?
    # I will use the 8x Strassen 2x2 approach for simplicity, rank 56.
    # Wait, 56 + 4 + 4 = 64.
    # Schoolbook is 100.
    # 64 is an improvement.
    
    # Let's try to get a lower rank for M_4,4,4.
    # If I use 4 independent 2x2 mults for the diagonal blocks and 0 for off-diagonal? No.
    
    # I will implement the 8x Strassen 2x2 decomposition for M_4,4,4.
    # Rank 56.
    
    # 2. M_4,4,1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # Rank 4.
    
    # 3. M_4,1,5: A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # Rank 4.
    
    # Total Rank 56 + 4 + 4 = 64.
    
    # Helper to add Strassen 2x2 decomposition
    def add_strassen_2x2(U, V, W, i_off, j_off, k_off, col_start):
        # A is 2x2 at i_off, j_off
        # B is 2x2 at j_off, k_off
        # C is 2x2 at k_off, i_off
        # Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # A11: (i_off, j_off), A12: (i_off, j_off+1)
        # A21: (i_off+1, j_off), A22: (i_off+1, j_off+1)
        
        # B11: (j_off, k_off), B12: (j_off, k_off+1)
        # B21: (j_off+1, k_off), B22: (j_off+1, k_off+1)
        
        # C11: (k_off, i_off), C12: (k_off, i_off+1)
        # C21: (k_off+1, i_off), C22: (k_off+1, i_off+1)
        
        def set_U(i, j, val):
            U[i * m + j][col_start] = val
        def set_V(j, k, val):
            V[j * p + k][col_start] = val
        def set_W(k, i, val):
            W[k * n + i][col_start] = val

        # M1
        set_U(i_off, j_off, 1); set_U(i_off+1, j_off+1, 1)
        set_V(j_off, k_off, 1); set_V(j_off+1, k_off+1, 1)
        set_W(k_off, i_off, 1); set_W(k_off+1, i_off+1, 1)
        col_start += 1

        # M2
        set_U(i_off+1, j_off, 1); set_U(i_off+1, j_off+1, 1)
        set_V(j_off, k_off, 1)
        set_W(k_off+1, i_off, 1); set_W(k_off+1, i_off+1, 1)
        col_start += 1

        # M3
        set_U(i_off, j_off, 1)
        set_V(j_off, k_off+1, 1); set_V(j_off+1, k_off+1, -1)
        set_W(k_off, i_off+1, 1); set_W(k_off+1, i_off, 1)
        col_start += 1

        # M4
        set_U(i_off+1, j_off+1, 1)
        set_V(j_off+1, k_off, 1); set_V(j_off, k_off, -1)
        set_W(k_off+1, i_off, 1); set_W(k_off, i_off+1, 1)
        col_start += 1

        # M5
        set_U(i_off, j_off, 1); set_U(i_off, j_off+1, 1)
        set_V(j_off+1, k_off+1, 1)
        set_W(k_off, i_off+1, -1); set_W(k_off+1, i_off+1, 1)
        col_start += 1

        # M6
        set_U(i_off+1, j_off, 1); set_U(i_off, j_off, -1)
        set_V(j_off, k_off, 1); set_V(j_off, k_off+1, 1)
        set_W(k_off+1, i_off+1, 1)
        col_start += 1

        # M7
        set_U(i_off, j_off+1, 1); set_U(i_off+1, j_off+1, -1)
        set_V(j_off+1, k_off, 1); set_V(j_off+1, k_off+1, 1)
        set_W(k_off, i_off, 1)
        col_start += 1

        return col_start

    # Apply 8 Strassen 2x2 for M_4,4,4
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    # A11(0,0), A12(0,1), A21(1,0), A22(1,1)
    # B11(0,0), B12(0,1), B21(1,0), B22(1,1)
    
    # C11 = A11 B11 + A12 B21
    col = add_strassen_2x2(U, V, W, 0, 0, 0, col) # A11 B11
    col = add_strassen_2x2(U, V, W, 0, 2, 2, col) # A12 B21 (A12 is 0,2? No. A is 4x4. A12 is rows 0-1, cols 2-3)
    
    # Wait, A is 4x4.
    # A11: rows 0-1, cols 0-1.
    # A12: rows 0-1, cols 2-3.
    # A21: rows 2-3, cols 0-1.
    # A22: rows 2-3, cols 2-3.
    
    # B11: rows 0-1, cols 0-1.
    # B12: rows 0-1, cols 2-3.
    # B21: rows 2-3, cols 0-1.
    # B22: rows 2-3, cols 2-3.
    
    # C11 (rows 0-1, cols 0-1) = A11 B11 + A12 B21
    # C12 (rows 0-1, cols 2-3) = A11 B12 + A12 B22
    # C21 (rows 2-3, cols 0-1) = A21 B11 + A22 B21
    # C22 (rows 2-3, cols 2-3) = A21 B12 + A22 B22
    
    # My add_strassen_2x2 takes i_off, j_off, k_off.
    # A is at i_off, j_off.
    # B is at j_off, k_off.
    # C is at k_off, i_off.
    
    # For A11 B11: i_off=0, j_off=0, k_off=0.
    col = add_strassen_2x2(U, V, W, 0, 0, 0, col)
    
    # For A12 B21:
    # A12: i_off=0, j_off=2.
    # B21: j_off=2, k_off=0. (B21 is rows 2-3, cols 0-1. So j_off=2, k_off=0)
    # C11: k_off=0, i_off=0.
    col = add_strassen_2x2(U, V, W, 0, 2, 0, col)
    
    # For A11 B12:
    # A11: i_off=0, j_off=0.
    # B12: j_off=0, k_off=2.
    # C12: k_off=2, i_off=0.
    col = add_strassen_2x2(U, V, W, 0, 0, 2, col)
    
    # For A12 B22:
    # A12: i_off=0, j_off=2.
    # B22: j_off=2, k_off=2.
    # C12: k_off=2, i_off=0.
    col = add_strassen_2x2(U, V, W, 0, 2, 2, col)
    
    # For A21 B11:
    # A21: i_off=2, j_off=0.
    # B11: j_off=0, k_off=0.
    # C21: k_off=0, i_off=2.
    col = add_strassen_2x2(U, V, W, 2, 0, 0, col)
    
    # For A22 B21:
    # A22: i_off=2, j_off=2.
    # B21: j_off=2, k_off=0.
    # C21: k_off=0, i_off=2.
    col = add_strassen_2x2(U, V, W, 2, 2, 0, col)
    
    # For A21 B12:
    # A21: i_off=2, j_off=0.
    # B12: j_off=0, k_off=2.
    # C22: k_off=2, i_off=2.
    col = add_strassen_2x2(U, V, W, 2, 0, 2, col)
    
    # For A22 B22:
    # A22: i_off=2, j_off=2.
    # B22: j_off=2, k_off=2.
    # C22: k_off=2, i_off=2.
    col = add_strassen_2x2(U, V, W, 2, 2, 2, col)
    
    # 2. M_4,4,1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # Rank 4.
    # C_i = sum_j A_i,j B_j,4
    # T = sum_i e_i (U) * 1_4 (V) * e_i (W)
    for i in range(4):
        U[i * m + 0][col] = 1
        # V is 5x5. j in 0..3. k=4.
        for j in range(4):
            V[j * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    # 3. M_4,1,5: A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # Rank 4.
    # C_i,k = A_i,4 B_4,k
    # T = sum_i e_i (U) * e_4 (V) * 1_5 (W)
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 0][col] = 1
        for k in range(5):
            W[k * n + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
