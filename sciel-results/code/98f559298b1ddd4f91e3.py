import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the rank 60 decomposition.
    # Total Rank = 60
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a sub-decomposition
    # sub_U, sub_V, sub_W are for a sub-problem with dimensions (sn, sm, sp)
    # offset_u, offset_v, offset_w are the starting row indices in the global matrices
    # rank_offset is the starting column index in the rank dimension
    def add_subproblem(sub_U, sub_V, sub_W, offset_u, offset_v, offset_w, rank_offset):
        for i in range(len(sub_U)):
            for c in range(len(sub_U[0])):
                U[offset_u + i][rank_offset + c] = sub_U[i][c]
        for i in range(len(sub_V)):
            for c in range(len(sub_V[0])):
                V[offset_v + i][rank_offset + c] = sub_V[i][c]
        for i in range(len(sub_W)):
            for c in range(len(sub_W[0])):
                W[offset_w + i][rank_offset + c] = sub_W[i][c]

    # 1. Generate Schoolbook for 1x3x7 (A2 * B)
    # sn=1, sm=3, sp=7
    sn, sm, sp = 1, 3, 7
    r1 = sn * sm * sp # 21
    U1 = [[0] * r1 for _ in range(sn * sm)]
    V1 = [[0] * r1 for _ in range(sm * sp)]
    W1 = [[0] * r1 for _ in range(sp * sn)]
    col = 0
    for i in range(sn):
        for k in range(sm):
            for j in range(sp):
                U1[i * sm + k][col] = 1
                V1[k * sp + j][col] = 1
                W1[j * sn + i][col] = 1
                col += 1
                
    # Map to global
    # A2 is row 3 (index 2 in 0-based, but task uses 1-based logic for indices? 
    # Task: U[i*m+j, c]. If i is 1-based, then i=3.
    # Python lists are 0-based.
    # If task means i=0..n-1, then A2 is i=2.
    # "U[i*m+j, c]" usually implies 0-based in code, but math is 1-based.
    # Let's assume 0-based indices for the matrix rows.
    # A2 corresponds to i=2.
    # U rows for i=2: 2*3 + k (k=0,1,2). So 6, 7, 8.
    offset_u1 = 2 * m
    offset_v1 = 0
    offset_w1 = 0 # W rows are j*n + i. i=2. So j*3 + 2.
    # Wait, W1 rows are j*sn + i. Here sn=1, so j*1 + 0 = j.
    # Global W index is j*n + i. Here n=3, i=2.
    # So global row is j*3 + 2.
    # sub_W row is j.
    # So we need to map sub_W[j] to W[j*3 + 2].
    
    # This mapping is tricky with the simple add_subproblem.
    # I will just fill the global matrices directly for each term.
    
    # Let's restart the filling logic.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Part 1: A2 * B (1x3x7)
    # i=2 (0-based)
    for k in range(3):
        for j in range(7):
            U[2 * m + k][current_rank] = 1
            V[k * p + j][current_rank] = 1
            W[j * n + 2][current_rank] = 1
            current_rank += 1
            
    # Part 2: A1 * B (2x3x7)
    # A1 is rows 0, 1.
    # Split B into B1 (3x3) and B2 (3x4).
    
    # Part 2a: A1 * B1 (2x3x3)
    # Split A1 into A11 (2x2) and A12 (2x1).
    # Split B1 into B11 (2x3) and B12 (1x3).
    
    # A11 * B11 (2x2x3)
    # Use Strassen for 2x2x2 part (cols 0,1 of B11) and Schoolbook for col 2 of B11.
    # B11 cols are 0,1,2.
    # Strassen on B11[:, 0:2]
    # Schoolbook on B11[:, 2]
    
    # Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices in A11: i=0,1; k=0,1.
    # Indices in B11: k=0,1; j=0,1.
    
    def add_strassen_term(U, V, W, rank_idx, a_coeffs, b_coeffs, c_coeffs, i_offset, k_offset, j_offset):
        # a_coeffs: list of (i, k, val)
        # b_coeffs: list of (k, j, val)
        # c_coeffs: list of (i, j, val)
        for (i, k, val) in a_coeffs:
            U[(i_offset + i) * m + (k_offset + k)][rank_idx] = val
        for (k, j, val) in b_coeffs:
            V[(k_offset + k) * p + (j_offset + j)][rank_idx] = val
        for (i, j, val) in c_coeffs:
            W[(j_offset + j) * n + (i_offset + i)][rank_idx] = val

    # Strassen Terms
    # 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)], 
                      0, 0, 0)
    current_rank += 1
    # 2
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (1,1,1)], 
                      [(0,0,1)], 
                      [(1,0,1)], 
                      0, 0, 0)
    current_rank += 1
    # 3
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1)], 
                      [(0,1,1), (1,1,-1)], 
                      [(0,1,1)], 
                      0, 0, 0)
    current_rank += 1
    # 4
    add_strassen_term(U, V, W, current_rank, 
                      [(1,1,1)], 
                      [(1,0,1), (0,0,-1)], 
                      [(1,0,1)], 
                      0, 0, 0)
    current_rank += 1
    # 5
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      [(0,1,1)], 
                      0, 0, 0)
    current_rank += 1
    # 6
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (0,0,-1)], 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      0, 0, 0)
    current_rank += 1
    # 7
    add_strassen_term(U, V, W, current_rank, 
                      [(0,1,1), (1,1,-1)], 
                      [(1,0,1), (1,1,1)], 
                      [(1,1,1)], 
                      0, 0, 0)
    current_rank += 1
    
    # Schoolbook for A11 * B11[:, 2] (2x2x1)
    # A11 (2x2), B11_col2 (2x1)
    for i in range(2):
        for k in range(2):
            U[i * m + k][current_rank] = 1
            V[k * p + 2][current_rank] = 1 # j=2
            W[2 * n + i][current_rank] = 1 # j=2, i
            current_rank += 1
            
    # A12 * B12 (2x1x3)
    # A12 is col 2 of A1 (k=2). i=0,1.
    # B12 is row 2 of B1 (k=2). j=0,1,2.
    for i in range(2):
        for j in range(3):
            U[i * m + 2][current_rank] = 1
            V[2 * p + j][current_rank] = 1
            W[j * n + i][current_rank] = 1
            current_rank += 1

    # Part 2b: A1 * B2 (2x3x4)
    # Split B2 into B21 (3x2) and B22 (3x2).
    # B2 cols are 3,4 and 5,6.
    
    # A1 * B21 (2x3x2)
    # Isomorphic to 2x2x3 (Rank 11)
    # A1 (2x3), B21 (3x2).
    # Split A1 into A11 (2x2) and A12 (2x1).
    # Split B21 into B21_top (2x2) and B21_bot (1x2).
    
    # A11 * B21_top (2x2x2) -> Strassen (Rank 7)
    # B21_top is rows 0,1 of B21. Cols 0,1 of B21.
    # Global B indices: k=0,1; j=3,4.
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (1,1,1)], 
                      [(0,0,1)], 
                      [(1,0,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1)], 
                      [(0,1,1), (1,1,-1)], 
                      [(0,1,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,1,1)], 
                      [(1,0,1), (0,0,-1)], 
                      [(1,0,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      [(0,1,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (0,0,-1)], 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      0, 0, 3)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,1,1), (1,1,-1)], 
                      [(1,0,1), (1,1,1)], 
                      [(1,1,1)], 
                      0, 0, 3)
    current_rank += 1
    
    # A11 * B21_bot (2x2x1) -> Schoolbook (Rank 4)
    # B21_bot is row 2 of B21. Cols 0,1 of B21.
    # Global B: k=2; j=3,4.
    for i in range(2):
        for k in range(2):
            for j in range(2):
                U[i * m + k][current_rank] = 1
                V[2 * p + (3 + j)][current_rank] = 1
                W[(3 + j) * n + i][current_rank] = 1
                current_rank += 1
                
    # A12 * B21 (2x1x2) -> Schoolbook (Rank 4)
    # A12 is k=2. i=0,1.
    # B21 is k=0,1,2. j=3,4.
    for i in range(2):
        for k in range(3):
            for j in range(2):
                U[i * m + 2][current_rank] = 1
                V[k * p + (3 + j)][current_rank] = 1
                W[(3 + j) * n + i][current_rank] = 1
                current_rank += 1

    # A1 * B22 (2x3x2)
    # Same as above but j_offset = 5
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,1)], 
                      [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (1,1,1)], 
                      [(0,0,1)], 
                      [(1,0,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1)], 
                      [(0,1,1), (1,1,-1)], 
                      [(0,1,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,1,1)], 
                      [(1,0,1), (0,0,-1)], 
                      [(1,0,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      [(0,1,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(1,0,1), (0,0,-1)], 
                      [(0,0,1), (0,1,1)], 
                      [(1,1,1)], 
                      0, 0, 5)
    current_rank += 1
    add_strassen_term(U, V, W, current_rank, 
                      [(0,1,1), (1,1,-1)], 
                      [(1,0,1), (1,1,1)], 
                      [(1,1,1)], 
                      0, 0, 5)
    current_rank += 1
    
    # A11 * B22_bot (2x2x1) -> Schoolbook (Rank 4)
    for i in range(2):
        for k in range(2):
            for j in range(2):
                U[i * m + k][current_rank] = 1
                V[2 * p + (5 + j)][current_rank] = 1
                W[(5 + j) * n + i][current_rank] = 1
                current_rank += 1
                
    # A12 * B22 (2x1x2) -> Schoolbook (Rank 4)
    for i in range(2):
        for k in range(3):
            for j in range(2):
                U[i * m + 2][current_rank] = 1
                V[k * p + (5 + j)][current_rank] = 1
                W[(5 + j) * n + i][current_rank] = 1
                current_rank += 1

    return {"U": U, "V": V, "W": W}
