import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # A indices: (i, j) in 2x2 block
    # B indices: (j, k) in 2x2 block
    # C indices: (i, k) in 2x2 block
    # We need to map local indices to global indices.
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each term is a rank 1 tensor? No, it's a product of two linear combinations.
        # In the tensor decomposition, each multiplication M_i corresponds to a rank 1 term?
        # No, M_i is a scalar product of two linear combinations of inputs.
        # So each M_i contributes 1 to the rank.
        
        # Term 1
        # U part: A11+A22
        # V part: B11+B22
        # W part: C11, C22 (with signs)
        
        # We need to add 7 terms.
        
        # T1: (A11+A22) * (B11+B22)
        U[a_row_offset + 0][a_col_offset + 0][col] += 1
        U[a_row_offset + 1][a_col_offset + 1][col] += 1
        V[b_row_offset + 0][b_col_offset + 0][col] += 1
        V[b_row_offset + 0][b_col_offset + 1][col] += 1
        W[c_col_offset + 0][c_row_offset + 0][col] += 1 # C11
        W[c_col_offset + 1][c_row_offset + 1][col] += 1 # C22
        col += 1
        
        # T2: (A21+A22) * B11
        U[a_row_offset + 1][a_col_offset + 0][col] += 1
        U[a_row_offset + 1][a_col_offset + 1][col] += 1
        V[b_row_offset + 0][b_col_offset + 0][col] += 1
        W[c_col_offset + 0][c_row_offset + 1][col] += 1 # C21
        W[c_col_offset + 0][c_row_offset + 1][col] += 1 # Wait, C21 is (1,0) in local?
        # Local C indices: i=0,1; k=0,1.
        # C21 is i=1, k=0.
        # W index is k*n + i.
        # Global W index: (c_col_offset + k) * n + (c_row_offset + i)
        
        # Let's refine the W indexing.
        # W[k*n + i, r]
        
        # Restart T2
        col -= 1 # Undo
        # T2: (A21+A22) * B11
        U[a_row_offset + 1][a_col_offset + 0][col] += 1
        U[a_row_offset + 1][a_col_offset + 1][col] += 1
        V[b_row_offset + 0][b_col_offset + 0][col] += 1
        # C21: i=1, k=0
        W[0 * n + 1][col] += 1 
        # Wait, W index is k*n + i.
        # k=0, i=1 -> 0*n + 1 = 1.
        # But we have offsets.
        # k_global = b_col_offset + 0
        # i_global = a_row_offset + 1
        # W index = (b_col_offset + 0) * n + (a_row_offset + 1)
        W[(b_col_offset + 0) * n + (a_row_offset + 1)][col] += 1
        col += 1
        
        # This manual mapping is error prone. I will use a more systematic approach.
        # I will build the matrices using numpy and then convert to list of lists.
        
    # Better approach: Use numpy to build U, V, W.
    U_np = np.zeros((n * m, rank))
    V_np = np.zeros((m * p, rank))
    W_np = np.zeros((p * n, rank))
    
    col = 0
    
    # 1. A_2x4 * B_4x4
    # Split A_2x4 into A_2x2_1 (cols 0-1), A_2x2_2 (cols 2-3)
    # Split B_4x4 into B_4x2_1 (cols 0-1), B_4x2_2 (cols 2-3)
    # Product 1: A_2x2_1 * B_4x2_1
    #   Split B_4x2_1 into B_2x2_11 (rows 0-1), B_2x2_12 (rows 2-3)
    #   Prod 1.1: A_2x2_1 * B_2x2_11 (Strassen)
    #   Prod 1.2: A_2x2_1 * B_2x2_12 (Strassen)
    # Product 2: A_2x2_2 * B_4x2_2
    #   Split B_4x2_2 into B_2x2_21 (rows 0-1), B_2x2_22 (rows 2-3)
    #   Prod 2.1: A_2x2_2 * B_2x2_21 (Strassen)
    #   Prod 2.2: A_2x2_2 * B_2x2_22 (Strassen)
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r, a_c: top-left of 2x2 A block
        # b_r, b_c: top-left of 2x2 B block
        # c_r, c_c: top-left of 2x2 C block
        
        # T1: (A11+A22)(B11+B22)
        U_np[a_r + a_c * 2, col] += 1 # A11? No, A is row-major?
        # A index: i*m + j.
        # A11: i=0, j=0. Index a_r*5 + a_c?
        # Wait, m=5.
        # A is 2x5.
        # A[i, j] index is i*5 + j.
        
        # Let's use indices directly.
        # A11: (a_r, a_c)
        # A22: (a_r+1, a_c+1)
        # B11: (b_r, b_c)
        # B22: (b_r+1, b_c+1)
        
        # T1
        U_np[a_r * m + a_c, col] += 1
        U_np[(a_r + 1) * m + (a_c + 1), col] += 1
        V_np[b_r * p + b_c, col] += 1
        V_np[(b_r + 1) * p + (b_c + 1), col] += 1
        # C11: (c_r, c_c) -> W[c_c * n + c_r]
        # C22: (c_r+1, c_c+1) -> W[(c_c+1) * n + (c_r+1)]
        W_np[c_c * n + c_r, col] += 1
        W_np[(c_c + 1) * n + (c_r + 1), col] += 1
        col += 1
        
        # T2: (A21+A22)B11
        U_np[(a_r + 1) * m + a_c, col] += 1
        U_np[(a_r + 1) * m + (a_c + 1), col] += 1
        V_np[b_r * p + b_c, col] += 1
        # C21: (c_r+1, c_c) -> W[c_c * n + (c_r+1)]
        W_np[c_c * n + (c_r + 1), col] += 1
        col += 1
        
        # T3: A11(B12-B22)
        U_np[a_r * m + a_c, col] += 1
        V_np[b_r * p + (b_c + 1), col] += 1
        V_np[(b_r + 1) * p + (b_c + 1), col] -= 1
        # C12: (c_r, c_c+1) -> W[(c_c+1) * n + c_r]
        W_np[(c_c + 1) * n + c_r, col] += 1
        col += 1
        
        # T4: A22(B21-B11)
        U_np[(a_r + 1) * m + (a_c + 1), col] += 1
        V_np[(b_r + 1) * p + b_c, col] += 1
        V_np[b_r * p + b_c, col] -= 1
        # C21: (c_r+1, c_c) -> W[c_c * n + (c_r+1)]
        W_np[c_c * n + (c_r + 1), col] += 1
        col += 1
        
        # T5: (A11+A12)B22
        U_np[a_r * m + a_c, col] += 1
        U_np[a_r * m + (a_c + 1), col] += 1
        V_np[(b_r + 1) * p + (b_c + 1), col] += 1
        # C11: (c_r, c_c) -> W[c_c * n + c_r]
        W_np[c_c * n + c_r, col] -= 1
        # C12: (c_r, c_c+1) -> W[(c_c+1) * n + c_r]
        W_np[(c_c + 1) * n + c_r, col] += 1
        col += 1
        
        # T6: (A21-A11)(B11+B12)
        U_np[(a_r + 1) * m + a_c, col] += 1
        U_np[a_r * m + a_c, col] -= 1
        V_np[b_r * p + b_c, col] += 1
        V_np[b_r * p + (b_c + 1), col] += 1
        # C22: (c_r+1, c_c+1) -> W[(c_c+1) * n + (c_r+1)]
        W_np[(c_c + 1) * n + (c_r + 1), col] += 1
        col += 1
        
        # T7: (A12-A22)(B21+B22)
        U_np[a_r * m + (a_c + 1), col] += 1
        U_np[(a_r + 1) * m + (a_c + 1), col] -= 1
        V_np[(b_r + 1) * p + b_c, col] += 1
        V_np[(b_r + 1) * p + (b_c + 1), col] += 1
        # C11: (c_r, c_c) -> W[c_c * n + c_r]
        W_np[c_c * n + c_r, col] += 1
        # C22: (c_r+1, c_c+1) -> W[(c_c+1) * n + (c_r+1)]
        W_np[(c_c + 1) * n + (c_r + 1), col] -= 1
        col += 1

    # Call for the 4 Strassen blocks in A_2x4 * B_4x4
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Wait, B is 4x4.
    # B[0:2, 0:2] is rows 0-1, cols 0-1.
    # B[2:4, 0:2] is rows 2-3, cols 0-1.
    # A[0:2, 0:2] is rows 0-1, cols 0-1.
    # Product is C[0:2, 0:2].
    add_strassen_block(0, 0, 2, 0, 0, 0)
    
    # Block 3: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 0, 2, 0, 2)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # 2. A_2x1 * B_1x4
    # A[0:2, 4] * B[4, 0:4] -> C[0:2, 0:4]
    # Rank 8
    for i in range(2):
        for k in range(4):
            U_np[i * m + 4, col] += 1
            V_np[4 * p + k, col] += 1
            W_np[k * n + i, col] += 1
            col += 1
            
    # 3. A_2x4 * B_4x1
    # A[0:2, 0:4] * B[0:4, 4] -> C[0:2, 4]
    # Rank 8
    for i in range(2):
        for j in range(4):
            U_np[i * m + j, col] += 1
            V_np[j * p + 4, col] += 1
            W_np[4 * n + i, col] += 1
            col += 1
            
    # 4. A_2x1 * B_1x1
    # A[0:2, 4] * B[4, 4] -> C[0:2, 4]
    # Rank 2
    for i in range(2):
        U_np[i * m + 4, col] += 1
        V_np[4 * p + 4, col] += 1
        W_np[4 * n + i, col] += 1
        col += 1
        
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

