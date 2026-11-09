"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    
    # We construct a rank 22 decomposition.
    # A (2x3) = [A_L (2x2), A_R (2x1)]
    # B (3x4) = [B_T (2x4); B_B (1x4)]
    # B_T = [B_T1 (2x2), B_T2 (2x2)]
    # C = A_L B_T1 + A_L B_T2 + A_R B_B
    
    # Indices for A: 0..5 (row-major 2x3)
    # i in 0..1, j in 0..2 -> idx = i*3 + j
    # A_L: j in 0,1. A_R: j in 2.
    
    # Indices for B: 0..11 (row-major 3x4)
    # j in 0..2, k in 0..3 -> idx = j*4 + k
    # B_T: j in 0,1. B_B: j in 2.
    # B_T1: k in 0,1. B_T2: k in 2,3.
    
    # Indices for C: 0..7 (row-major 2x4)
    # i in 0..1, k in 0..3 -> idx = i*4 + k
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 6 x 22
    V = [[0] * rank for _ in range(m * p)] # 12 x 22
    W = [[0] * rank for _ in range(p * n)] # 8 x 22
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # A_sub (2x2) from A indices, B_sub (2x2) from B indices
    # Output C_sub (2x2) to C indices
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A indices: A_rows are i, A_cols are j
        # B indices: B_rows are j, B_cols are k
        # C indices: C_rows are i, C_cols are k
        
        # Strassen 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to map these to the global indices.
        # A11: i=A_rows[0], j=A_cols[0]
        # A12: i=A_rows[0], j=A_cols[1]
        # A21: i=A_rows[1], j=A_cols[0]
        # A22: i=A_rows[1], j=A_cols[1]
        
        # B11: j=B_rows[0], k=B_cols[0]
        # B12: j=B_rows[0], k=B_cols[1]
        # B21: j=B_rows[1], k=B_cols[0]
        # B22: j=B_rows[1], k=B_cols[1]
        
        # C11: i=C_rows[0], k=C_cols[0]
        # C12: i=C_rows[0], k=C_cols[1]
        # C21: i=C_rows[1], k=C_cols[0]
        # C22: i=C_rows[1], k=C_cols[1]
        
        def idx_U(i, j): return i * m + j
        def idx_V(j, k): return j * p + k
        def idx_W(k, i): return k * n + i
        
        # M1
        U[idx_U(A_rows[0], A_cols[0])][col] += 1
        U[idx_U(A_rows[1], A_cols[1])][col] += 1
        V[idx_V(B_rows[0], B_cols[0])][col] += 1
        V[idx_V(B_rows[1], B_cols[1])][col] += 1
        W[idx_W(C_cols[0], C_rows[0])][col] += 1
        W[idx_W(C_cols[1], C_rows[0])][col] += 1
        W[idx_W(C_cols[0], C_rows[1])][col] += 1
        W[idx_W(C_cols[1], C_rows[1])][col] += 1
        col += 1
        
        # M2
        U[idx_U(A_rows[1], A_cols[0])][col] += 1
        U[idx_U(A_rows[1], A_cols[1])][col] += 1
        V[idx_V(B_rows[0], B_cols[0])][col] += 1
        W[idx_W(C_cols[0], C_rows[1])][col] += 1
        W[idx_W(C_cols[1], C_rows[1])][col] += 1
        col += 1
        
        # M3
        U[idx_U(A_rows[0], A_cols[0])][col] += 1
        V[idx_V(B_rows[1], B_cols[0])][col] += 1
        V[idx_V(B_rows[1], B_cols[1])][col] += -1
        W[idx_W(C_cols[1], C_rows[0])][col] += 1
        col += 1
        
        # M4
        U[idx_U(A_rows[1], A_cols[1])][col] += 1
        V[idx_V(B_rows[0], B_cols[1])][col] += 1
        V[idx_V(B_rows[0], B_cols[0])][col] += -1
        W[idx_W(C_cols[1], C_rows[1])][col] += 1
        col += 1
        
        # M5
        U[idx_U(A_rows[0], A_cols[0])][col] += 1
        U[idx_U(A_rows[0], A_cols[1])][col] += 1
        V[idx_V(B_rows[1], B_cols[1])][col] += 1
        W[idx_W(C_cols[1], C_rows[0])][col] += 1
        W[idx_W(C_cols[1], C_rows[1])][col] += 1
        col += 1
        
        # M6
        U[idx_U(A_rows[1], A_cols[0])][col] += 1
        U[idx_U(A_rows[0], A_cols[0])][col] += -1
        V[idx_V(B_rows[0], B_cols[0])][col] += 1
        V[idx_V(B_rows[0], B_cols[1])][col] += 1
        W[idx_W(C_cols[0], C_rows[1])][col] += 1
        col += 1
        
        # M7
        U[idx_U(A_rows[0], A_cols[1])][col] += 1
        U[idx_U(A_rows[1], A_cols[1])][col] += -1
        V[idx_V(B_rows[1], B_cols[0])][col] += 1
        V[idx_V(B_rows[1], B_cols[1])][col] += 1
        W[idx_W(C_cols[0], C_rows[0])][col] += 1
        col += 1

    # Part 1: A_L B_T1
    # A_L: rows 0,1; cols 0,1
    # B_T1: rows 0,1; cols 0,1
    # C: rows 0,1; cols 0,1
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Part 2: A_L B_T2
    # A_L: rows 0,1; cols 0,1
    # B_T2: rows 0,1; cols 2,3
    # C: rows 0,1; cols 2,3
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Part 3: A_R B_B
    # A_R: rows 0,1; col 2
    # B_B: row 2; cols 0,1,2,3
    # C: rows 0,1; cols 0,1,2,3
    # C[i][k] = A[0][2]*B[2][k] for i=0
    # C[i][k] = A[1][2]*B[2][k] for i=1
    
    # i=0, k=0
    U[idx_U(0, 2)][col] = 1
    V[idx_V(2, 0)][col] = 1
    W[idx_W(0, 0)][col] = 1
    col += 1
    
    # i=0, k=1
    U[idx_U(0, 2)][col] = 1
    V[idx_V(2, 1)][col] = 1
    W[idx_W(1, 0)][col] = 1
    col += 1
    
    # i=0, k=2
    U[idx_U(0, 2)][col] = 1
    V[idx_V(2, 2)][col] = 1
    W[idx_W(2, 0)][col] = 1
    col += 1
    
    # i=0, k=3
    U[idx_U(0, 2)][col] = 1
    V[idx_V(2, 3)][col] = 1
    W[idx_W(3, 0)][col] = 1
    col += 1
    
    # i=1, k=0
    U[idx_U(1, 2)][col] = 1
    V[idx_V(2, 0)][col] = 1
    W[idx_W(0, 1)][col] = 1
    col += 1
    
    # i=1, k=1
    U[idx_U(1, 2)][col] = 1
    V[idx_V(2, 1)][col] = 1
    W[idx_W(1, 1)][col] = 1
    col += 1
    
    # i=1, k=2
    U[idx_U(1, 2)][col] = 1
    V[idx_V(2, 2)][col] = 1
    W[idx_W(2, 1)][col] = 1
    col += 1
    
    # i=1, k=3
    U[idx_U(1, 2)][col] = 1
    V[idx_V(2, 3)][col] = 1
    W[idx_W(3, 1)][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
