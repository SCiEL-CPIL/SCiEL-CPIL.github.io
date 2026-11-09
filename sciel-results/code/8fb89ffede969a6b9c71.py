import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (30, rank)  
    # W: (p*n, rank) = (18, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (idx, val), etc."""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Process 3 blocks of (2,5,2) and 3 blocks of (1,5,2)
    # A rows: [0,1] for first group, [2] for second group
    # B col groups: [0,1], [2,3], [4,5]
    
    for col_group in range(3):
        c1 = col_group * 2
        c2 = c1 + 1  # columns c1, c2
        
        # (2,5,2) block: A rows 0,1; B cols c1,c2
        # Split middle: j in [0,1] (Strassen part) and j in [2,3,4] (naive part)
        
        # Strassen part: (2,2,2)
        # A sub: rows [0,1], cols [0,1] -> indices in A: (0,0),(0,1),(1,0),(1,1)
        # B sub: rows [0,1], cols [c1,c2] -> indices in B: (0,c1),(0,c2),(1,c1),(1,c2)
        # C sub: rows [0,1], cols [c1,c2] -> indices in C: (0,c1),(0,c2),(1,c1),(1,c2)
        
        # Strassen M1 = (A00+A11)(B00+B11)
        # In our indexing: A[0,0]+A[1,1] times B[0,c1]+B[0,c2]... 
        # Wait, I need to be more careful.
        
        # For the (2,2,2) sub-problem:
        # A_local[0,0] = A[0,0], A_local[0,1] = A[0,1]
        # A_local[1,0] = A[1,0], A_local[1,1] = A[1,1]
        # B_local[0,0] = B[0,c1], B_local[0,1] = B[0,c2]
        # B_local[1,0] = B[1,c1], B_local[1,1] = B[1,c2]
        # C_local[0,0] = C[0,c1], C_local[0,1] = C[0,c2]
        # C_local[1,0] = C[1,c1], C_local[1,1] = C[1,c2]
        
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear combination of A entries and a linear combination of B entries.
        # For the tensor decomposition, each "term" in Strassen corresponds to one rank-1 component
        # where U encodes the A coefficients, V encodes the B coefficients, and W encodes the C contributions.
        
        # M1: A coeff: A[0,0]=1, A[1,1]=1; B coeff: B[0,c1]=1, B[0,c2]=1... 
        # Wait, B00+B11 means B[0,c1] + B[1,c2]. 
        # Hmm, B_local[0,0]=B[0,c1], B_local[1,1]=B[1,c2]
        # So B00+B11 = B[0,c1]+B[1,c2]
        # And A00+A11 = A[0,0]+A[1,1]
        # M1 contributes to C00 and C11.
        
        # Actually, in the bilinear decomposition, each Strassen multiplication Mi corresponds to
        # a single rank-1 term where:
        # U has the A coefficients for that Mi
        # V has the B coefficients for that Mi  
        # W has the C coefficients (which Mi contributes to, with what sign)
        
        # M1: U: A[0,0]=1, A[1,1]=1
        #      V: B[0,c1]=1, B[1,c2]=1
        #      W: C[0,c1]=1, C[1,c2]=1  (M1 appears in C00 and C11)
        
        # Wait, C00 = M1 + M4 - M5 + M7, so M1 contributes +1 to C[0,c1]
        # C11 = M1 - M2 + M3 + M6, so M1 contributes +1 to C[1,c2]
        
        # Let me define helper for Strassen on a (2,2,2) block
        def strassen_222(a_row1, a_row2, b_col1, b_col2, c_row1, c_row2):
            """
            a_row1, a_row2: rows of A (0 and 1 in local, mapped to global)
            b_col1, b_col2: cols of B
            c_row1, c_row2: rows of C
            The middle dimension for this sub-block is j in [0,1] (first two of the 5)
            """
            # A indices: (a_row1, 0), (a_row1, 1), (a_row2, 0), (a_row2, 1)
            # B indices: (0, b_col1), (0, b_col2), (1, b_col1), (1, b_col2)
            # C indices: (c_row1, b_col1), (c_row1, b_col2), (c_row2, b_col1), (c_row2, b_col2)
            
            A00 = (a_row1, 0)  # A[a_row1, 0]
            A01 = (a_row1, 1)  # A[a_row1, 1]
            A10 = (a_row2, 0)  # A[a_row2, 0]
            A11 = (a_row2, 1)  # A[a_row2, 1]
            
            B00 = (0, b_col1)   # B[0, b_col1]
            B01 = (0, b_col2)   # B[0, b_col2]
            B10 = (1, b_col1)   # B[1, b_col1]
            B11 = (1, b_col2)   # B[1, b_col2]
            
            C00 = (c_row1, b_col1)  # C[c_row1, b_col1]
            C01 = (c_row1, b_col2)  # C[c_row1, b_col2]
            C10 = (c_row2, b_col1)  # C[c_row2, b_col1]
            C11 = (c_row2, b_col2)  # C[c_row2, b_col2]
            
            def a_idx(i, j): return i * m + j
            def b_idx(i, j): return i * p + j
            def c_idx(i, j): return j * n + i  # W[k*n+i] where k=col, i=row
            
            # M1 = (A00+A11)(B00+B11)
            add_term(
                [(a_idx(*A00), 1), (a_idx(*A11), 1)],
                [(b_idx(*B00), 1), (b_idx(*B11), 1)],
                [(c_idx(*C00), 1), (c_idx(*C11), 1)]
            )
            # M2 = (A10+A11)B00
            add_term(
                [(a_idx(*A10), 1), (a_idx(*A11), 1)],
                [(b_idx(*B00), 1)],
                [(c_idx(*C10), 1)]
            )
            # M3 = A00(B10-B11)
            add_term(
                [(a_idx(*A00), 1)],
                [(b_idx(*B10), 1), (b_idx(*B11), -1)],
                [(c_idx(*C01), 1)]
            )
            # M4 = A11(B01-B00)
            add_term(
                [(a_idx(*A11), 1)],
                [(b_idx(*B01), 1), (b_idx(*B00), -1)],
                [(c_idx(*C00), 1), (c_idx(*C10), 1)]
            )
            # M5 = (A00+A10)B11
            add_term(
                [(a_idx(*A00), 1), (a_idx(*A10), 1)],
                [(b_idx(*B11), 1)],
                [(c_idx(*C00), -1), (c_idx(*C01), 1)]
            )
            # M6 = (A10-A00)(B00+B10)
            add_term(
                [(a_idx(*A10), 1), (a_idx(*A00), -1)],
                [(b_idx(*B00), 1), (b_idx(*B10), 1)],
                [(c_idx(*C11), 1)]
            )
            # M7 = (A01-A11)(B01+B11)
            add_term(
                [(a_idx(*A01), 1), (a_idx(*A11), -1)],
                [(b_idx(*B01), 1), (b_idx(*B11), 1)],
                [(c_idx(*C00), 1), (c_idx(*C11), 1)]
            )
        
        # Naive part for (2,3,2): A rows [0,1], cols [2,3,4]; B rows [2,3,4], cols [c1,c2]
        def naive_232(a_row1, a_row2, b_col1, b_col2, c_row1, c_row2):
            for j in range(2, 5):
                for ar in [a_row1, a_row2]:
                    for bc in [b_col1, b_col2]:
                        add_term(
                            [(ar * m + j, 1)],
                            [(j * p + bc, 1)],
                            [(bc * n + ar, 1)]
                        )
        
        # Apply Strassen for the (2,2,2) part (j=0,1)
        strassen_222(0, 1, c1, c2, 0, 1)
        # Apply naive for the (2,3,2) part (j=2,3,4)
        naive_232(0, 1, c1, c2, 0, 1)
        
        # (1,5,2) block: A row 2; B cols c1,c2
        # Naive: for each j in 0..4, for each col in [c1,c2]
        for j in range(5):
            for bc in [c1, c2]:
                add_term(
                    [(2 * m + j, 1)],
                    [(j * p + bc, 1)],
                    [(bc * n + 2, 1)]
                )
    
    rank = len(U)
    # Convert to lists (already are)
    return {"U": U, "V": V, "W": W}
