"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zero_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]
        
    # Helper to add two matrices of same shape
    def add_mat(A, B):
        rows = len(A)
        cols = len(A[0])
        C = zero_mat(rows, cols)
        for i in range(rows):
            for j in range(cols):
                C[i][j] = A[i][j] + B[i][j]
        return C

    # Helper to scale a matrix
    def scale_mat(A, s):
        rows = len(A)
        cols = len(A[0])
        C = zero_mat(rows, cols)
        for i in range(rows):
            for j in range(cols):
                C[i][j] = A[i][j] * s
        return C

    # Decomposition for M_{2,2,2} using Strassen (Rank 7)
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1 # C11+=M1, C12+=M1, C21-=M1, C22+=M1? 
        # Let's use the standard Strassen formulas:
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
        
        # Reset W
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Decomposition for M_{2,2,1} using Schoolbook (Rank 4)
    def schoolbook_221():
        # A: 2x2, B: 2x1, C: 2x1
        # C1 = A11*B11 + A21*B21
        # C2 = A12*B12 + A22*B22
        # Note: B is 2x1. Indices: 0:(0,0), 1:(1,0).
        # C is 2x1. Indices: 0:(0,0), 1:(1,0).
        # W index convention: k*n + i. Here p=1, n=2.
        # k=0. i=0 -> idx 0. i=1 -> idx 1.
        
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # Term 1: A11 * B11 -> C1
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 2: A21 * B21 -> C1
        U[2][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # Term 3: A12 * B12 -> C2
        # Wait, B is 2x1. B12 doesn't exist.
        # B is 2x1. Rows 0,1. Col 0.
        # A is 2x2.
        # C is 2x1.
        # C[0,0] = A[0,0]B[0,0] + A[1,0]B[1,0]
        # C[1,0] = A[0,1]B[0,0] + A[1,1]B[1,0]
        
        # My previous assumption about B indices was wrong for 2x1.
        # B indices: 0:(0,0), 1:(1,0).
        
        # Term 3: A01 * B00 -> C1
        U[1][2] = 1
        V[0][2] = 1
        W[1][2] = 1
        
        # Term 4: A11 * B10 -> C1
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W

    # Assemble M_{2,4,7}
    # Split B (4x7) into 4 parts: 4x2, 4x2, 4x2, 4x1
    # For each part, split A (2x4) into 2x2, 2x2
    # And split B_part (4xk) into 2xk, 2xk
    
    U_total = []
    V_total = []
    W_total = []
    
    def append_block(U_block, V_block, W_block, a_offset, b_offset, c_offset):
        # a_offset: row offset in A (0 or 2)
        # b_offset: row offset in B (0 or 2)
        # c_offset: col offset in C (0, 2, 4, 6)
        
        n_b = len(V_block)
        n_w = len(W_block)
        rank_b = len(V_block[0])
        
        # Map U indices
        # U_block is for 2x2 A.
        # A indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global A indices:
        # If a_offset=0: 0->0, 1->1, 2->2, 3->3
        # If a_offset=2: 0->2, 1->3, 2->4, 3->5
        
        # Map V indices
        # V_block is for 2xk B.
        # B indices in block: j*p + k.
        # If b_offset=0: 0->0, 1->1, 2->2, 3->3 (for k=2)
        # If b_offset=2: 0->2, 1->3, 2->4, 3->5 (for k=2)
        
        # Map W indices
        # W_block is for 2xk C.
        # W indices in block: k*n + i.
        # Global C indices: k_global * 2 + i.
        # k_global = c_offset + k_local.
        
        for r in range(rank_b):
            u_col = [0]*8 # n*m = 2*4 = 8
            v_col = [0]*28 # m*p = 4*7 = 28
            w_col = [0]*14 # p*n = 7*2 = 14
            
            for i in range(4):
                # U mapping
                if a_offset == 0:
                    u_idx = i
                else:
                    u_idx = i + 2
                u_col[u_idx] = U_block[i][r]
                
            for i in range(n_b):
                # V mapping
                # V_block row i corresponds to B row (i // k) and col (i % k)?
                # No, V_block is for 2xk.
                # Indices in V_block: 0..2k-1.
                # If k=2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
                # If b_offset=0: global idx = i.
                # If b_offset=2: global idx = i + 2*k.
                
                # We need to know k.
                # For 2x2 block, k=2. n_b=4.
                # For 2x1 block, k=1. n_b=2.
                
                k_val = n_b // 2
                if b_offset == 0:
                    v_idx = i
                else:
                    v_idx = i + 2 * k_val
                v_col[v_idx] = V_block[i][r]
                
            for i in range(n_w):
                # W mapping
                # W_block row i corresponds to C col (i // 2) and row (i % 2)?
                # W index convention: k*n + i.
                # For 2xk C.
                # Indices: 0..2k-1.
                # i = k_local * 2 + row_local.
                # k_local = i // 2.
                # row_local = i % 2.
                # Global k = c_offset + k_local.
                # Global W idx = Global k * 2 + row_local.
                
                k_local = i // 2
                row_local = i % 2
                global_k = c_offset + k_local
                w_idx = global_k * 2 + row_local
                w_col[w_idx] = W_block[i][r]
                
            U_total.append(u_col)
            V_total.append(v_col)
            W_total.append(w_col)

    # Block 1: C cols 0,1. B cols 0,1.
    # A_L (0,1) * B_T (0,1)
    U1, V1, W1 = strassen_222()
    append_block(U1, V1, W1, 0, 0, 0)
    # A_R (2,3) * B_B (2,3)
    U2, V2, W2 = strassen_222()
    append_block(U2, V2, W2, 2, 2, 0)
    
    # Block 2: C cols 2,3. B cols 2,3.
    U3, V3, W3 = strassen_222()
    append_block(U3, V3, W3, 0, 0, 2)
    U4, V4, W4 = strassen_222()
    append_block(U4, V4, W4, 2, 2, 2)
    
    # Block 3: C cols 4,5. B cols 4,5.
    U5, V5, W5 = strassen_222()
    append_block(U5, V5, W5, 0, 0, 4)
    U6, V6, W6 = strassen_222()
    append_block(U6, V6, W6, 2, 2, 4)
    
    # Block 4: C col 6. B col 6.
    U7, V7, W7 = schoolbook_221()
    append_block(U7, V7, W7, 0, 0, 6)
    U8, V8, W8 = schoolbook_221()
    append_block(U8, V8, W8, 2, 2, 6)
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
