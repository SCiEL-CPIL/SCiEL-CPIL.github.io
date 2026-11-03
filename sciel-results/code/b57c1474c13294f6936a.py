"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build a decomposition of rank 52.
    # M_{2,4,7} -> M_{2,2,7} (x2)
    # M_{2,2,7} -> M_{2,2,4} (rank 14) + M_{2,2,3} (rank 12)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # We are building columns.
        # This approach is messy. Better to build U, V, W as lists of columns.
        pass

    # Let's build U, V, W as lists of lists (rows).
    # U: 8 rows. V: 28 rows. W: 14 rows.
    # Rank 52.
    
    U = [[0]*52 for _ in range(8)]
    V = [[0]*52 for _ in range(28)]
    W = [[0]*52 for _ in range(14)]
    
    col = 0
    
    # Strassen for M_{2,2,2}
    # Inputs: A (2x2), B (2x2). Output C (2x2).
    # U factors (for A):
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    # V factors (for B):
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    # W factors (for C):
    # M1: C00 + C11
    # M2: C10
    # M3: C01
    # M4: C10
    # M5: C01
    # M6: C11
    # M7: C00
    # Wait, W is for the output tensor.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the W vectors for the 7 terms are:
    # Term 1: (1, 0, 0, 1) for C00, C01, C10, C11?
    # W index is k*n + i.
    # C is 2x2. n=2.
    # C00 -> idx 0
    # C01 -> idx 1
    # C10 -> idx 2
    # C11 -> idx 3
    # W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, 0]
    # W3 = [0, 1, 0, 0]
    # W4 = [0, 0, 1, 0]
    # W5 = [0, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    # W7 = [1, 0, 0, 0]
    
    # Let's define the Strassen components
    # A indices in local 2x2: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # U1 = [1, 0, 0, 1]
    # U2 = [0, 0, 1, 1]
    # U3 = [1, 0, 0, 0]
    # U4 = [0, 0, 0, 1]
    # U5 = [1, 1, 0, 0]
    # U6 = [-1, 0, 1, 0]
    # U7 = [0, 1, 0, -1]
    
    # B indices in local 2x2: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # V1 = [1, 0, 0, 1]
    # V2 = [1, 0, 0, 0]
    # V3 = [0, -1, 1, 0]
    # V4 = [-1, 1, 0, 0]
    # V5 = [0, 0, 0, 1]
    # V6 = [1, 1, 0, 0]
    # V7 = [0, 1, 1, 0]
    
    # W vectors (size 4)
    # W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, 0]
    # W3 = [0, 1, 0, 0]
    # W4 = [0, 0, 1, 0]
    # W5 = [0, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    # W7 = [1, 0, 0, 0]
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Function to add a block of M_{2,2,2} to the global U,V,W
    # a_start: row offset in A (0 or 2)
    # b_start: row offset in B (0 or 2) AND col offset in B (0,2,4,6)
    # c_start: col offset in C (0,2,4,6)
    # Note: B is 4x7. B rows are j=0..3. B cols are k=0..6.
    # A is 2x4. A rows i=0..1. A cols j=0..3.
    # C is 2x7. C rows i=0..1. C cols k=0..6.
    
    def add_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        nonlocal col
        # a_col_offset: 0 for A1, 2 for A2.
        # b_row_offset: 0 for B1, 2 for B2.
        # b_col_offset: 0 for B11, 2 for B12? No, B11 is cols 0-3.
        # For M_{2,2,4}, we split cols into 0-1 and 2-3.
        # So b_col_offset can be 0 or 2.
        # c_col_offset: 0 or 2.
        
        for t in range(7):
            # U
            for local_i in range(2):
                local_j = t # No, U_strassen[t] is a vector of size 4.
                pass
            
            # U_strassen[t] has 4 entries corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
            # Global A indices:
            # A[0, a_col_offset]
            # A[0, a_col_offset+1]
            # A[1, a_col_offset]
            # A[1, a_col_offset+1]
            
            u_vec = U_strassen[t]
            # u_vec[0] -> A[0, a_col_offset] -> global idx 0*4 + a_col_offset
            # u_vec[1] -> A[0, a_col_offset+1] -> global idx 0*4 + a_col_offset+1
            # u_vec[2] -> A[1, a_col_offset] -> global idx 1*4 + a_col_offset
            # u_vec[3] -> A[1, a_col_offset+1] -> global idx 1*4 + a_col_offset+1
            
            U[0*4 + a_col_offset][col] += u_vec[0]
            U[0*4 + a_col_offset+1][col] += u_vec[1]
            U[1*4 + a_col_offset][col] += u_vec[2]
            U[1*4 + a_col_offset+1][col] += u_vec[3]
            
            # V
            # V_strassen[t] has 4 entries corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
            # Global B indices:
            # B[b_row_offset, b_col_offset]
            # B[b_row_offset, b_col_offset+1]
            # B[b_row_offset+1, b_col_offset]
            # B[b_row_offset+1, b_col_offset+1]
            
            v_vec = V_strassen[t]
            V[b_row_offset*7 + b_col_offset][col] += v_vec[0]
            V[b_row_offset*7 + b_col_offset+1][col] += v_vec[1]
            V[(b_row_offset+1)*7 + b_col_offset][col] += v_vec[2]
            V[(b_row_offset+1)*7 + b_col_offset+1][col] += v_vec[3]
            
            # W
            # W_strassen[t] has 4 entries corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
            # Global C indices (W index is k*n + i):
            # C[0, c_col_offset] -> idx c_col_offset*2 + 0
            # C[0, c_col_offset+1] -> idx (c_col_offset+1)*2 + 0
            # C[1, c_col_offset] -> idx c_col_offset*2 + 1
            # C[1, c_col_offset+1] -> idx (c_col_offset+1)*2 + 1
            
            w_vec = W_strassen[t]
            W[c_col_offset*2 + 0][col] += w_vec[0]
            W[(c_col_offset+1)*2 + 0][col] += w_vec[1]
            W[c_col_offset*2 + 1][col] += w_vec[2]
            W[(c_col_offset+1)*2 + 1][col] += w_vec[3]
            
            col += 1

    # Function to add a standard block of M_{2,2,3}
    def add_standard_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x3, C is 2x3
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U
                    U[i*4 + (a_col_offset + j)][col] = 1
                    # V
                    V[(b_row_offset + j)*7 + (b_col_offset + k)][col] = 1
                    # W
                    W[(c_col_offset + k)*2 + i][col] = 1
                    col += 1

    # 1. A1 B11 (M_{2,2,4} part 1: cols 0-1)
    # A1 is a_col_offset 0. B11 is b_row_offset 0, b_col_offset 0. C11 is c_col_offset 0.
    # We need two Strassen blocks for M_{2,2,4}: one for cols 0-1, one for cols 2-3.
    add_strassen_block(0, 0, 0, 0) # Cols 0-1
    add_strassen_block(0, 0, 2, 2) # Cols 2-3
    
    # 2. A1 B12 (M_{2,2,3} part: cols 4-6)
    # A1 is a_col_offset 0. B12 is b_row_offset 0, b_col_offset 4. C12 is c_col_offset 4.
    add_standard_block(0, 0, 4, 4)
    
    # 3. A2 B21 (M_{2,2,4} part 1: cols 0-1)
    # A2 is a_col_offset 2. B21 is b_row_offset 2, b_col_offset 0. C21 is c_col_offset 0.
    add_strassen_block(2, 2, 0, 0)
    add_strassen_block(2, 2, 2, 2)
    
    # 4. A2 B22 (M_{2,2,3} part: cols 4-6)
    # A2 is a_col_offset 2. B22 is b_row_offset 2, b_col_offset 4. C22 is c_col_offset 4.
    add_standard_block(2, 2, 4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
