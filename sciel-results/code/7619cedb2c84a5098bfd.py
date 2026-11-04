"""
Program to find the minimal rank tensor decomposition for 3x4 * 4x5 matrix multiplication.
Approach: Use block decomposition with Strassen's algorithm for 2x2x2 blocks and
schoolbook for smaller blocks, then combine to get a rank-minimal exact decomposition.
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    
    Uses block decomposition:
    - Split A (3x4) into blocks: top 2 rows, bottom 1 row
    - Split B (4x5) into blocks: first 2 cols, next 2 cols, last 1 col
    - This gives 3x2 = 6 block multiplications
    
    For each block multiplication (n_i x m_j) * (m_j x p_k):
    - If it's 2x2x2, use Strassen (rank 7)
    - If it's 2x2x1, use schoolbook (rank 4)
    - If it's 2x1x2, use schoolbook (rank 4)
    - If it's 1x2x2, use schoolbook (rank 4)
    - If it's 2x1x1, use schoolbook (rank 2)
    - If it's 1x2x1, use schoolbook (rank 2)
    - If it's 1x1x2, use schoolbook (rank 2)
    - If it's 1x1x1, use schoolbook (rank 1)
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Define block splits
    # A is 3x4: split into [2x2, 2x2, 1x2, 1x2] (row splits: 2,1; col splits: 2,2)
    # B is 4x5: split into [2x2, 2x2, 2x1, 2x2, 2x2, 2x1] (row splits: 2,2; col splits: 2,2,1)
    
    # Row splits for A (n=3): [2, 1]
    # Col splits for A (m=4): [2, 2]
    # Row splits for B (m=4): [2, 2]
    # Col splits for B (p=5): [2, 2, 1]
    
    a_row_splits = [2, 1]
    a_col_splits = [2, 2]
    b_row_splits = [2, 2]
    b_col_splits = [2, 2, 1]
    
    # For each combination of (a_row_idx, b_col_idx), we have a block multiplication
    # A_block is (a_row_splits[a_row_idx]) x (a_col_splits[j] for some j)
    # Actually, let's think more carefully.
    
    # The full multiplication is C = A * B where A is 3x4, B is 4x5, C is 3x5.
    # We split A into rows: [0:2], [2:3]
    # We split B into columns: [0:2], [2:4], [4:5]
    # And we split the shared dimension (4) into: [0:2], [2:4]
    
    # So we have blocks:
    # For row block r in [0:2], [2:3]
    # For col block c in [0:2], [2:4], [4:5]
    # For mid block j in [0:2], [2:4]
    #
    # C[r, c] = sum_j A[r, j] * B[j, c]
    
    # Let's define the indices more carefully.
    # A has rows 0,1,2 and cols 0,1,2,3
    # B has rows 0,1,2,3 and cols 0,1,2,3,4
    # C has rows 0,1,2 and cols 0,1,2,3,4
    
    # Block structure:
    # A rows: block 0 = [0,1], block 1 = [2]
    # A cols: block 0 = [0,1], block 1 = [2,3]
    # B rows: block 0 = [0,1], block 1 = [2,3]
    # B cols: block 0 = [0,1], block 1 = [2,3], block 2 = [4]
    
    # For each (row_block, col_block), we compute C[row_block, col_block]
    # which is sum over mid_block j of A[row_block, j] * B[j, col_block]
    
    # So:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   where A[0,0] is 2x2, B[0,0] is 2x2 -> 2x2x2, rank 7 (Strassen)
    #   where A[0,1] is 2x2, B[1,0] is 2x2 -> 2x2x2, rank 7 (Strassen)
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   where A[0,0] is 2x2, B[0,1] is 2x2 -> 2x2x2, rank 7 (Strassen)
    #   where A[0,1] is 2x2, B[1,1] is 2x2 -> 2x2x2, rank 7 (Strassen)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   where A[0,0] is 2x2, B[0,2] is 2x1 -> 2x2x1, rank 4 (schoolbook)
    #   where A[0,1] is 2x2, B[1,2] is 2x1 -> 2x2x1, rank 4 (schoolbook)
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   where A[1,0] is 1x2, B[0,0] is 2x2 -> 1x2x2, rank 4 (schoolbook)
    #   where A[1,1] is 1x2, B[1,0] is 2x2 -> 1x2x2, rank 4 (schoolbook)
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   where A[1,0] is 1x2, B[0,1] is 2x2 -> 1x2x2, rank 4 (schoolbook)
    #   where A[1,1] is 1x2, B[1,1] is 2x2 -> 1x2x2, rank 4 (schoolbook)
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #   where A[1,0] is 1x2, B[0,2] is 2x1 -> 1x2x1, rank 2 (schoolbook)
    #   where A[1,1] is 1x2, B[1,2] is 2x1 -> 1x2x1, rank 2 (schoolbook)
    
    # Total rank:
    # C[0,0]: 7 + 7 = 14
    # C[0,1]: 7 + 7 = 14
    # C[0,2]: 4 + 4 = 8
    # C[1,0]: 4 + 4 = 8
    # C[1,1]: 4 + 4 = 8
    # C[1,2]: 2 + 2 = 4
    # Total: 14 + 14 + 8 + 8 + 8 + 4 = 56
    
    # Hmm, that's not as good as I thought. Let me reconsider.
    
    # Actually, I think I can do better by using a different block structure.
    # Let me try: A rows [2,1], A cols [2,2], B rows [2,2], B cols [2,2,1]
    # But the key insight is that for 2x2x2, Strassen gives rank 7 instead of 8.
    
    # Let me just implement the straightforward approach and see what rank we get.
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        """Add Strassen decomposition for 2x2x2 block."""
        # Strassen's algorithm for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        global rank
        local_rank = 7
        rank += local_rank
        
        for t in range(7):
            u_row = [0] * (3 * 4)
            v_row = [0] * (4 * 5)
            w_row = [0] * (5 * 3)
            
            if t == 0:
                # M1 = (a+d)(e+h)
                # a = A[a_row_start, a_col_start], d = A[a_row_start+1, a_col_start+1]
                # e = B[b_row_start, b_col_start], h = B[b_row_start+1, b_col_start+1]
                u_row[a_row_start * 4 + a_col_start] = 1
                u_row[(a_row_start+1) * 4 + (a_col_start+1)] = 1
                v_row[b_row_start * 5 + b_col_start] = 1
                v_row[(b_row_start+1) * 5 + (b_col_start+1)] = 1
                # C contribution: M1 goes to C[c_row_start, c_col_start] with +1
                # and to C[c_row_start, c_col_start+1] with +1? No, let me check.
                # Actually, in Strassen:
                # C[0,0] = M1 + M4 - M5 + M7
                # C[0,1] = M3 + M5
                # C[1,0] = M2 + M4
                # C[1,1] = M1 - M2 + M3 + M6
                # So M1 contributes to C[0,0] with +1 and C[1,1] with +1
                w_row[c_col_start * 3 + c_row_start] = 1
                w_row[(c_col_start+1) * 3 + (c_row_start+1)] = 1
                
            elif t == 1:
                # M2 = (c+d)e
                # c = A[a_row_start+1, a_col_start], d = A[a_row_start+1, a_col_start+1]
                # e = B[b_row_start, b_col_start]
                u_row[(a_row_start+1) * 4 + a_col_start] = 1
                u_row[(a_row_start+1) * 4 + (a_col_start+1)] = 1
                v_row[b_row_start * 5 + b_col_start] = 1
                # M2 contributes to C[1,0] with +1 and C[1,1] with -1
                w_row[c_col_start * 3 + (c_row_start+1)] = 1
                w_row[(c_col_start+1) * 3 + (c_row_start+1)] = -1
                
            elif t == 2:
                # M3 = a(f-h)
                # a = A[a_row_start, a_col_start]
                # f = B[b_row_start, b_col_start+1], h = B[b_row_start+1, b_col_start+1]
                u_row[a_row_start * 4 + a_col_start] = 1
                v_row[b_row_start * 5 + (b_col_start+1)] = 1
                v_row[(b_row_start+1) * 5 + (b_col_start+1)] = -1
                # M3 contributes to C[0,1] with +1 and C[1,1] with +1
                w_row[(c_col_start+1) * 3 + c_row_start] = 1
                w_row[(c_col_start+1) * 3 + (c_row_start+1)] = 1
                
            elif t == 3:
                # M4 = d(g-e)
                # d = A[a_row_start+1, a_col_start+1]
                # g = B[b_row_start+1, b_col_start], e = B[b_row_start, b_col_start]
                u_row[(a_row_start+1) * 4 + (a_col_start+1)] = 1
                v_row[(b_row_start+1) * 5 + b_col_start] = 1
                v_row[b_row_start * 5 + b_col_start] = -1
                # M4 contributes to C[0,0] with +1 and C[1,0] with +1
                w_row[c_col_start * 3 + c_row_start] = 1
                w_row[c_col_start * 3 + (c_row_start+1)] = 1
                
            elif t == 4:
                # M5 = (a+b)h
                # a = A[a_row_start, a_col_start], b = A[a_row_start, a_col_start+1]
                # h = B[b_row_start+1, b_col_start+1]
                u_row[a_row_start * 4 + a_col_start] = 1
                u_row[a_row_start * 4 + (a_col_start+1)] = 1
                v_row[(b_row_start+1) * 5 + (b_col_start+1)] = 1
                # M5 contributes to C[0,0] with -1 and C[0,1] with +1
                w_row[c_col_start * 3 + c_row_start] = -1
                w_row[(c_col_start+1) * 3 + c_row_start] = 1
                
            elif t == 5:
                # M6 = (c-a)(e+f)
                # c = A[a_row_start+1, a_col_start], a = A[a_row_start, a_col_start]
                # e = B[b_row_start, b_col_start], f = B[b_row_start, b_col_start+1]
                u_row[(a_row_start+1) * 4 + a_col_start] = 1
                u_row[a_row_start * 4 + a_col_start] = -1
                v_row[b_row_start * 5 + b_col_start] = 1
                v_row[b_row_start * 5 + (b_col_start+1)] = 1
                # M6 contributes to C[1,1] with +1
                w_row[(c_col_start+1) * 3 + (c_row_start+1)] = 1
                
            elif t == 6:
                # M7 = (b-d)(g+h)
                # b = A[a_row_start, a_col_start+1], d = A[a_row_start+1, a_col_start+1]
                # g = B[b_row_start+1, b_col_start], h = B[b_row_start+1, b_col_start+1]
                u_row[a_row_start * 4 + (a_col_start+1)] = 1
                u_row[(a_row_start+1) * 4 + (a_col_start+1)] = -1
                v_row[(b_row_start+1) * 5 + b_col_start] = 1
                v_row[(b_row_start+1) * 5 + (b_col_start+1)] = 1
                # M7 contributes to C[0,0] with +1
                w_row[c_col_start * 3 + c_row_start] = 1
            
            U_rows.append(u_row)
            V_rows.append(v_row)
            W_rows.append(w_row)
    
    def add_schoolbook_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        """Add schoolbook decomposition for 2x2x2 block (rank 8)."""
        global rank
        local_rank = 8
        rank += local_rank
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_row = [0] * (3 * 4)
                    v_row = [0] * (4 * 5)
                    w_row = [0] * (5 * 3)
                    
                    u_row[(a_row_start + i) * 4 + (a_col_start + j)] = 1
                    v_row[(b_row_start + j) * 5 + (b_col_start + k)] = 1
                    w_row[(c_col_start + k) * 3 + (c_row_start + i)] = 1
                    
                    U_rows.append(u_row)
                    V_rows.append(v_row)
                    W_rows.append(w_row)
    
    def add_schoolbook_2x2x1(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        """Add schoolbook decomposition for 2x2x1 block (rank 4)."""
        global rank
        local_rank = 4
        rank += local_rank
        
        for i in range(2):
            for j in range(2):
                u_row = [0] * (3 * 4)
                v_row = [0] * (4 * 5)
                w_row = [0] * (5 * 3)
                
                u_row[(a_row_start + i) * 4 + (a_col_start + j)] = 1
                v_row[(b_row_start + j) * 5 + b_col_start] = 1
                w_row[c_col_start * 3 + (c_row_start + i)] = 1
                
                U_rows.append(u_row)
                V_rows.append(v_row)
                W_rows.append(w_row)
    
    def add_schoolbook_1x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        """Add schoolbook decomposition for 1x2x2 block (rank 4)."""
        global rank
        local_rank = 4
        rank += local_rank
        
        for j in range(2):
            for k in range(2):
                u_row = [0] * (3 * 4)
                v_row = [0] * (4 * 5)
                w_row = [0] * (5 * 3)
                
                u_row[a_row_start * 4 + (a_col_start + j)] = 1
                v_row[(b_row_start + j) * 5 + (b_col_start + k)] = 1
                w_row[(c_col_start + k) * 3 + c_row_start] = 1
                
                U_rows.append(u_row)
                V_rows.append(v_row)
                W_rows.append(w_row)
    
    def add_schoolbook_1x2x1(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        """Add schoolbook decomposition for 1x2x1 block (rank 2)."""
        global rank
        local_rank = 2
        rank += local_rank
        
        for j in range(2):
            u_row = [0] * (3 * 4)
            v_row = [0] * (4 * 5)
            w_row = [0] * (5 * 3)
            
            u_row[a_row_start * 4 + (a_col_start + j)] = 1
            v_row[(b_row_start + j) * 5 + b_col_start] = 1
            w_row[c_col_start * 3 + c_row_start] = 1
            
            U_rows.append(u_row)
            V_rows.append(v_row)
            W_rows.append(w_row)
    
    # Now add all block decompositions
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # A[0,0] is 2x2 (rows 0-1, cols 0-1), B[0,0] is 2x2 (rows 0-1, cols 0-1)
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0, 0)
    # A[0,1] is 2x2 (rows 0-1, cols 2-3), B[1,0] is 2x2 (rows 2-3, cols 0-1)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0, 0)
    
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # A[0,0] is 2x2, B[0,1] is 2x2 (rows 0-1, cols 2-3)
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2, 0)
    # A[0,1] is 2x2, B[1,1] is 2x2 (rows 2-3, cols 2-3)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2, 0)
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # A[0,0] is 2x2, B[0,2] is 2x1 (rows 0-1, col 4)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, 0)
    # A[0,1] is 2x2, B[1,2] is 2x1 (rows 2-3, col 4)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, 0)
    
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # A[1,0] is 1x2 (row 2, cols 0-1), B[0,0] is 2x2 (rows 0-1, cols 0-1)
    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0, 0)
    # A[1,1] is 1x2 (row 2, cols 2-3), B[1,0] is 2x2 (rows 2-3, cols 0-1)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0, 0)
    
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # A[1,0] is 1x2, B[0,1] is 2x2 (rows 0-1, cols 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2, 0)
    # A[1,1] is 1x2, B[1,1] is 2x2 (rows 2-3, cols 2-3)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2, 0)
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # A[1,0] is 1x2, B[0,2] is 2x1 (rows 0-1, col 4)
    add_schoolbook_1x2x1(2, 0, 0, 4, 2, 4, 0)
    # A[1,1] is 1x2, B[1,2] is 2x1 (rows 2-3, col 4)
    add_schoolbook_1x2x1(2, 2, 2, 4, 2, 4, 0)
    
    # The global 'rank' variable should now hold the total rank
    total_rank = len(U_rows)
    
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
