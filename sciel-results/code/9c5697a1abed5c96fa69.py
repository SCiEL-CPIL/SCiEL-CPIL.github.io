<<<<<<< SEARCH
    # Partition: A rows [0,2), [2,3), A cols [0,2), [2,4)
    # B rows [0,2), [2,4), B cols [0,2), [2,4), [4,5)
    blocks = [
        (0, 2, 0, 2, 0, 2, 0, 2),
        (0, 2, 0, 2, 0, 2, 2, 4),
        (0, 2, 0, 2, 0, 2, 4, 5),
        (0, 2, 2, 4, 2, 4, 0, 2),
        (0, 2, 2, 4, 2, 4, 2, 4),
        (0, 2, 2, 4, 2, 4, 4, 5),
        (2, 3, 0, 2, 0, 2, 0, 2),
        (2, 3, 0, 2, 0, 2, 2, 4),
        (2, 3, 0, 2, 0, 2, 4, 5),
        (2, 3, 2, 4, 2, 4, 0, 2),
        (2, 3, 2, 4, 2, 4, 2, 4),
        (2, 3, 2, 4, 2, 4, 4, 5),
    ]
    
    for block in blocks:
        add_block(*block)
=======
    # Apply Strassen's algorithm to 2x2x2 blocks where possible
    # Strassen: rank 7 instead of 8 for 2x2x2 multiplication
    def add_strassen_block(a_r0, b_c0):
        """Add Strassen decomposition for a 2x2x2 block at position (a_r0, b_c0)"""
        nonlocal U, V, W
        if time.time() > deadline:
            return
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # Result: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        
        # Each Strassen product corresponds to one rank-1 term
        # We need to map these to our tensor structure
        # A is 2x2 (rows a_r0..a_r0+1, cols j..j+1 for some j)
        # B is 2x2 (rows j..j+1, cols b_c0..b_c0+1)
        
        # The 7 Strassen terms with their coefficient patterns:
        # For each term t, we have:
        # U contribution: linear combo of A elements
        # V contribution: linear combo of B elements  
        # W contribution: linear combo of C elements
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
            # V: coefficients for B[0,0], B[0,1], B[1,0], B[1,1]
            # W: coefficients for C[0,0], C[0,1], C[1,0], C[1,1]
            ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, -1]),   # M1: (A11+A22)(B11+B22) -> C11+=M1, C22-=M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),     # M2: (A21+A22)B11 -> C21+=M2, C22-=M2
            ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]),     # M3: A11(B21-B22) -> C12+=M3, C22+=M3
            ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 0, 0]),    # M4: A22(B12-B11) -> C11+=M4, C21+=M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),     # M5: (A11+A12)B22 -> C12+=M5, C11-=M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),     # M6: (A21-A11)(B11+B12) -> C22+=M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),     # M7: (A12-A22)(B21+B22) -> C11+=M7
        ]
        
        rank = 7
        U_new = np.zeros((n * m, rank))
        V_new = np.zeros((m * p, rank))
        W_new = np.zeros((p * n, rank))
        
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            # U: A is 2x2, indices (a_r0, j), (a_r0, j+1), (a_r0+1, j), (a_r0+1, j+1)
            # In our flattened U: row i, col j -> index i*m + j
            U_new[a_r0 * m + 0, t] = u_coeffs[0]
            U_new[a_r0 * m + 1, t] = u_coeffs[1]
            U_new[(a_r0 + 1) * m + 0, t] = u_coeffs[2]
            U_new[(a_r0 + 1) * m + 1, t] = u_coeffs[3]
            
            # V: B is 2x2, indices (j, b_c0), (j, b_c0+1), (j+1, b_c0), (j+1, b_c0+1)
            # In our flattened V: row j, col k -> index j*p + k
            V_new[0 * p + b_c0, t] = v_coeffs[0]
            V_new[0 * p + b_c0 + 1, t] = v_coeffs[1]
            V_new[1 * p + b_c0, t] = v_coeffs[2]
            V_new[1 * p + b_c0 + 1, t] = v_coeffs[3]
            
            # W: C is 2x2, indices (b_c0, a_r0), (b_c0, a_r0+1), (b_c0+1, a_r0), (b_c0+1, a_r0+1)
            # In our flattened W: row k, col i -> index k*n + i
            W_new[b_c0 * n + a_r0, t] = w_coeffs[0]
            W_new[b_c0 * n + a_r0 + 1, t] = w_coeffs[1]
            W_new[(b_c0 + 1) * n + a_r0, t] = w_coeffs[2]
            W_new[(b_c0 + 1) * n + a_r0 + 1, t] = w_coeffs[3]
        
        U = np.hstack([U, U_new])
        V = np.hstack([V, V_new])
        W = np.hstack([W, W_new])
    
    # Use Strassen for 2x2x2 blocks, standard for remaining
    # A rows: [0,2), [2,3)
    # A cols: [0,2), [2,4)
    # B rows: [0,2), [2,4)
    # B cols: [0,2), [2,4), [4,5)
    
    # For A rows [0,2) and A cols [0,2), B rows [0,2), B cols [0,2): Strassen
    add_strassen_block(0, 0)
    # For A rows [0,2) and A cols [0,2), B rows [0,2), B cols [2,4): Strassen
    add_strassen_block(0, 2)
    # For A rows [0,2) and A cols [0,2), B rows [0,2), B cols [4,5): standard (1x1x2)
    add_block(0, 2, 0, 2, 0, 2, 4, 5)
    
    # For A rows [0,2) and A cols [2,4), B rows [2,4), B cols [0,2): Strassen
    add_strassen_block(0, 0)  # This is wrong, need to handle offset properly
    
    # Let me redo this more carefully
    # The Strassen block operates on a 2x2 A subblock and 2x2 B subblock
    # where A subblock is at rows [a_r0, a_r0+2) and cols [j, j+2)
    # B subblock is at rows [j, j+2) and cols [b_c0, b_c0+2)
    # But in our problem, A cols and B rows are the same dimension (m=4)
    # So we need j to be consistent
    
    # Actually, let me reconsider the structure. The tensor is T[i,j,k] = delta_{j,k}
    # U[i,j,t], V[j,k,t], W[k,i,t]
    # For a Strassen block on A rows [r0,r0+2), A cols [c0,c0+2), B rows [c0,c0+2), B cols [k0,k0+2):
    # This works when A cols range = B rows range = [c0, c0+2)
    
    # Reset and redo
    U = np.zeros((n * m, 0), dtype=float)
    V = np.zeros((m * p, 0), dtype=float)
    W = np.zeros((p * n, 0), dtype=float)
    
    # Define Strassen for specific (a_row_start, a_col_start, b_col_start)
    def add_strassen(a_row_start, a_col_start, b_col_start):
        """Strassen for A[a_row_start:a_row_start+2, a_col_start:a_col_start+2] 
           times B[a_col_start:a_col_start+2, b_col_start:b_col_start+2]"""
        nonlocal U, V, W
        if time.time() > deadline:
            return
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        rank = 7
        U_new = np.zeros((n * m, rank))
        V_new = np.zeros((m * p, rank))
        W_new = np.zeros((p * n, rank))
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            # A indices: (a_row_start, a_col_start), (a_row_start, a_col_start+1),
            #            (a_row_start+1, a_col_start), (a_row_start+1, a_col_start+1)
            U_new[a_row_start * m + a_col_start, t] = u_coeffs[0]
            U_new[a_row_start * m + a_col_start + 1, t] = u_coeffs[1]
            U_new[(a_row_start + 1) * m + a_col_start, t] = u_coeffs[2]
            U_new[(a_row_start + 1) * m + a_col_start + 1, t] = u_coeffs[3]
            
            # B indices: (a_col_start, b_col_start), (a_col_start, b_col_start+1),
            #            (a_col_start+1, b_col_start), (a_col_start+1, b_col_start+1)
            V_new[a_col_start * p + b_col_start, t] = v_coeffs[0]
            V_new[a_col_start * p + b_col_start + 1, t] = v_coeffs[1]
            V_new[(a_col_start + 1) * p + b_col_start, t] = v_coeffs[2]
            V_new[(a_col_start + 1) * p + b_col_start + 1, t] = v_coeffs[3]
            
            # C indices: (b_col_start, a_row_start), (b_col_start, a_row_start+1),
            #            (b_col_start+1, a_row_start), (b_col_start+1, a_row_start+1)
            W_new[b_col_start * n + a_row_start, t] = w_coeffs[0]
            W_new[b_col_start * n + a_row_start + 1, t] = w_coeffs[1]
            W_new[(b_col_start + 1) * n + a_row_start, t] = w_coeffs[2]
            W_new[(b_col_start + 1) * n + a_row_start + 1, t] = w_coeffs[3]
        
        U = np.hstack([U, U_new])
        V = np.hstack([V, V_new])
        W = np.hstack([W, W_new])
    
    # Now apply Strassen where we have 2x2x2 blocks:
    # A rows [0,2), A cols [0,2), B rows [0,2), B cols [0,2)
    add_strassen(0, 0, 0)
    # A rows [0,2), A cols [0,2), B rows [0,2), B cols [2,4)
    add_strassen(0, 0, 2)
    # A rows [0,2), A cols [2,4), B rows [2,4), B cols [0,2)
    add_strassen(0, 2, 0)
    # A rows [0,2), A cols [2,4), B rows [2,4), B cols [2,4)
    add_strassen(0, 2, 2)
    
    # Remaining blocks (not 2x2x2):
    # A rows [0,2), A cols [0,2), B rows [0,2), B cols [4,5): 2x1x2 = 4 terms
    add_block(0, 2, 0, 2, 0, 2, 4, 5)
    # A rows [0,2), A cols [2,4), B rows [2,4), B cols [4,5): 2x2x1 = 4 terms
    add_block(0, 2, 2, 4, 2, 4, 4, 5)
    # A rows [2,3), A cols [0,2), B rows [0,2), B cols [0,2): 1x2x2 = 4 terms
    add_block(2, 3, 0, 2, 0, 2, 0, 2)
    # A rows [2,3), A cols [0,2), B rows [0,2), B cols [2,4): 1x2x2 = 4 terms
    add_block(2, 3, 0, 2, 0, 2, 2, 4)
    # A rows [2,3), A cols [0,2), B rows [0,2), B cols [4,5): 1x2x1 = 2 terms
    add_block(2, 3, 0, 2, 0, 2, 4, 5)
    # A rows [2,3), A cols [2,4), B rows [2,4), B cols [0,2): 1x2x2 = 4 terms
    add_block(2, 3, 2, 4, 2, 4, 0, 2)
    # A rows [2,3), A cols [2,4), B rows [2,4), B cols [2,4): 1x2x2 = 4 terms
    add_block(2, 3, 2, 4, 2, 4, 2, 4)
    # A rows [2,3), A cols [2,4), B rows [2,4), B cols [4,5): 1x2x1 = 2 terms
    add_block(2, 3, 2, 4, 2, 4, 4, 5)
>>>>>>> REPLACE
