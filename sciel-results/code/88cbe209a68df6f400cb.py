import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We will build U, V, W as lists of lists or numpy arrays.
    # Using lists for construction, then convert to numpy or keep as lists.
    # The return type should be {'U': U, 'V': V, 'W': W} with U, V, W as lists of lists or similar.
    
    # Start with empty lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off):
        """
        Adds a block decomposition to the global lists.
        U_block: list of rows, each row is a list of values for the block's U.
        V_block: list of rows for V.
        W_block: list of rows for W.
        i_off: row offset in A (and C)
        j_off: col offset in A (and row offset in B)
        k_off: col offset in B (and row offset in C)
        
        The block is at A[i_off:i_off+rows_A, j_off:j_off+cols_A]
        B[j_off:j_off+rows_B, k_off:k_off+cols_B]
        C[i_off:i_off+rows_C, k_off:k_off+cols_C]
        
        U_block has shape (rows_A * cols_A, rank_block)
        V_block has shape (rows_B * cols_B, rank_block)
        W_block has shape (rows_C * cols_C, rank_block)
        """
        rank_block = len(U_block[0]) if U_block else 0
        for c in range(rank_block):
            # Create full column vectors
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for r in range(len(U_block)):
                val = U_block[r][c]
                if val != 0:
                    # Map local row r to global row
                    # r is index in (rows_A * cols_A)
                    # local_row = r // cols_A, local_col = r % cols_A
                    local_row = r // cols_A
                    local_col = r % cols_A
                    global_row = (i_off + local_row) * m + (j_off + local_col)
                    u_col[global_row] = val
                    
            # Fill V
            for r in range(len(V_block)):
                val = V_block[r][c]
                if val != 0:
                    local_row = r // cols_B
                    local_col = r % cols_B
                    global_row = (j_off + local_row) * p + (k_off + local_col)
                    v_col[global_row] = val
                    
            # Fill W
            for r in range(len(W_block)):
                val = W_block[r][c]
                if val != 0:
                    local_row = r // cols_C
                    local_col = r % cols_C
                    global_row = (k_off + local_row) * n + (i_off + local_col)
                    w_col[global_row] = val
                    
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Define Strassen 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U rows: a, b, c, d (indices 0,1,2,3)
    # V rows: e, f, g, h (indices 0,1,2,3)
    # W rows: c11, c12, c21, c22 (indices 0,1,2,3)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U for M1: a=1, d=1, others 0. V: e=1, h=1.
    # We need to construct U_block, V_block, W_block for each M.
    
    def get_strassen_blocks():
        U = []
        V = []
        W = []
        # M1
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        W.append([1, 0, 0, 1]) # C11, C22
        # M2
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, -1]) # C21, C22
        # M3
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 1]) # C12, C22
        # M4
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        W.append([1, 0, 1, 0]) # C11, C21
        # M5
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([-1, 1, 0, 0]) # C11, C12
        # M6
        U.append([0, 0, 1, -1])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, 1]) # C22
        # M7
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([1, 0, 0, 0]) # C11
        
        return U, V, W

    U_str, V_str, W_str = get_strassen_blocks()
    
    # 1x2x2 (rank 4)
    # A = [a, b], B = [[e, f], [g, h]]
    # C = [ae+bg, af+bh]
    # U: a, b
    # V: e, f, g, h
    # W: c1, c2
    def get_1x2x2_blocks():
        U = []
        V = []
        W = []
        # ae
        U.append([1, 0])
        V.append([1, 0, 0, 0])
        W.append([1, 0])
        # bg
        U.append([0, 1])
        V.append([0, 0, 1, 0])
        W.append([1, 0])
        # af
        U.append([1, 0])
        V.append([0, 1, 0, 0])
        W.append([0, 1])
        # bh
        U.append([0, 1])
        V.append([0, 0, 0, 1])
        W.append([0, 1])
        return U, V, W

    U_122, V_122, W_122 = get_1x2x2_blocks()
    
    # 2x2x1 (rank 4)
    # A = [[a, b], [c, d]], B = [e, g]
    # C = [ae+bg, ce+dg]
    # U: a, b, c, d
    # V: e, g
    # W: c1, c2
    def get_2x2x1_blocks():
        U = []
        V = []
        W = []
        # ae
        U.append([1, 0, 0, 0])
        V.append([1, 0])
        W.append([1, 0])
        # bg
        U.append([0, 1, 0, 0])
        V.append([0, 1])
        W.append([1, 0])
        # ce
        U.append([0, 0, 1, 0])
        V.append([1, 0])
        W.append([0, 1])
        # dg
        U.append([0, 0, 0, 1])
        V.append([0, 1])
        W.append([0, 1])
        return U, V, W

    U_221, V_221, W_221 = get_2x2x1_blocks()
    
    # 1x2x1 (rank 2)
    # A = [a, b], B = [e, g]
    # C = ae+bg
    # U: a, b
    # V: e, g
    # W: c
    def get_1x2x1_blocks():
        U = []
        V = []
        W = []
        # ae
        U.append([1, 0])
        V.append([1, 0])
        W.append([1])
        # bg
        U.append([0, 1])
        V.append([0, 1])
        W.append([1])
        return U, V, W

    U_121, V_121, W_121 = get_1x2x1_blocks()

    # Now add blocks
    # Block 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2] -> 2x2x2 Strassen
    add_block(U_str, V_str, W_str, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2] -> 2x2x2 Strassen
    # Note: This contributes to C[0:2, 0:2] as well.
    # Wait, C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2]
    # So we add another 2x2x2 decomposition here.
    add_block(U_str, V_str, W_str, 0, 2, 0)
    
    # Block 3: A[0:2, 0:2], B[0:2, 2:4], C[0:2, 2:4] -> 2x2x2 Strassen
    add_block(U_str, V_str, W_str, 0, 0, 2)
    
    # Block 4: A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4] -> 2x2x2 Strassen
    add_block(U_str, V_str, W_str, 0, 2, 2)
    
    # Block 5: A[2:3, 0:2], B[0:2, 0:2], C[2:3, 0:2] -> 1x2x2
    add_block(U_122, V_122, W_122, 2, 0, 0)
    
    # Block 6: A[2:3, 2:4], B[2:4, 0:2], C[2:3, 0:2] -> 1x2x2
    add_block(U_122, V_122, W_122, 2, 2, 0)
    
    # Block 7: A[2:3, 0:2], B[0:2, 2:4], C[2:3, 2:4] -> 1x2x2
    add_block(U_122, V_122, W_122, 2, 0, 2)
    
    # Block 8: A[2:3, 2:4], B[2:4, 2:4], C[2:3, 2:4] -> 1x2x2
    add_block(U_122, V_122, W_122, 2, 2, 2)
    
    # Block 9: A[0:2, 0:2], B[0:2, 4:5], C[0:2, 4:5] -> 2x2x1
    add_block(U_221, V_221, W_221, 0, 0, 4)
    
    # Block 10: A[0:2, 2:4], B[2:4, 4:5], C[0:2, 4:5] -> 2x2x1
    add_block(U_221, V_221, W_221, 0, 2, 4)
    
    # Block 11: A[2:3, 0:2], B[0:2, 4:5], C[2:3, 4:5] -> 1x2x1
    add_block(U_121, V_121, W_121, 2, 0, 4)
    
    # Block 12: A[2:3, 2:4], B[2:4, 4:5], C[2:3, 4:5] -> 1x2x1
    add_block(U_121, V_121, W_121, 2, 2, 4)

    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
