import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # i in {1, 2} corresponds to rows 0-1, 2-3 of A and C
    # j in {1, 2, 3} corresponds to cols 0-1, 2-3, 4 of A and rows 0-1, 2-3, 4 of B
    # k in {1, 2, 3} corresponds to cols 0-1, 2-3, 4 of B and C
    
    # Block sizes
    # i=1: rows 0,1 (size 2)
    # i=2: rows 2,3 (size 2)
    
    # j=1: cols/rows 0,1 (size 2)
    # j=2: cols/rows 2,3 (size 2)
    # j=3: cols/rows 4 (size 1)
    
    # k=1: cols 0,1 (size 2)
    # k=2: cols 2,3 (size 2)
    # k=3: cols 4 (size 1)
    
    # We will build U, V, W lists of lists.
    # U shape: (n*m, rank) -> (20, rank)
    # V shape: (m*p, rank) -> (25, rank)
    # W shape: (p*n, rank) -> (20, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, sign_u, sign_v, sign_w):
        # u_idx, v_idx, w_idx are lists of indices
        # This adds a single column to U, V, W
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for idx in u_idx:
            u_col[idx] = sign_u
        for idx in v_idx:
            v_col[idx] = sign_v
        for idx in w_idx:
            w_col[idx] = sign_w
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Helper to get global indices for a block
    def get_A_indices(i_block, j_block):
        # i_block in {1, 2}, j_block in {1, 2, 3}
        # i_block 1 -> rows 0,1
        # i_block 2 -> rows 2,3
        # j_block 1 -> cols 0,1
        # j_block 2 -> cols 2,3
        # j_block 3 -> col 4
        
        rows = [0, 1] if i_block == 1 else [2, 3]
        if j_block == 1:
            cols = [0, 1]
        elif j_block == 2:
            cols = [2, 3]
        else:
            cols = [4]
            
        indices = []
        for r in rows:
            for c in cols:
                indices.append(r * m + c)
        return indices

    def get_B_indices(j_block, k_block):
        # j_block in {1, 2, 3}, k_block in {1, 2, 3}
        # j_block 1 -> rows 0,1
        # j_block 2 -> rows 2,3
        # j_block 3 -> row 4
        # k_block 1 -> cols 0,1
        # k_block 2 -> cols 2,3
        # k_block 3 -> col 4
        
        if j_block == 1:
            rows = [0, 1]
        elif j_block == 2:
            rows = [2, 3]
        else:
            rows = [4]
            
        if k_block == 1:
            cols = [0, 1]
        elif k_block == 2:
            cols = [2, 3]
        else:
            cols = [4]
            
        indices = []
        for r in rows:
            for c in cols:
                indices.append(r * p + c)
        return indices

    def get_C_indices(i_block, k_block):
        # i_block in {1, 2}, k_block in {1, 2, 3}
        # i_block 1 -> rows 0,1
        # i_block 2 -> rows 2,3
        # k_block 1 -> cols 0,1
        # k_block 2 -> cols 2,3
        # k_block 3 -> col 4
        
        rows = [0, 1] if i_block == 1 else [2, 3]
        if k_block == 1:
            cols = [0, 1]
        elif k_block == 2:
            cols = [2, 3]
        else:
            cols = [4]
            
        indices = []
        for r in rows:
            for c in cols:
                indices.append(c * n + r) # W[k*n+i, r] -> index is k*n + i
        return indices

    # 1. Strassen blocks for i in {1,2}, j in {1,2}, k in {1,2}
    for i_block in [1, 2]:
        for j_block in [1, 2]:
            for k_block in [1, 2]:
                # This is a 2x2x2 sub-problem
                # Local indices 0,1
                # A_block indices: get_A_indices(i_block, j_block)
                # But we need to map local (r, c) to global.
                # The 2x2 block A_{ij} has local indices (0,0), (0,1), (1,0), (1,1)
                # which correspond to global rows/cols.
                
                # We need to define the 7 Strassen terms for this specific block.
                # Let A_loc be 2x2, B_loc be 2x2, C_loc be 2x2.
                # We need to extract the specific global indices for A_loc[r,c], B_loc[r,c], C_loc[r,c].
                
                # Get global indices for the 2x2 block
                # A indices
                A_idx = get_A_indices(i_block, j_block)
                # A_idx is [r0c0, r0c1, r1c0, r1c1]
                # Local (0,0) -> A_idx[0]
                # Local (0,1) -> A_idx[1]
                # Local (1,0) -> A_idx[2]
                # Local (1,1) -> A_idx[3]
                
                B_idx = get_B_indices(j_block, k_block)
                # B_idx is [r0c0, r0c1, r1c0, r1c1]
                
                C_idx = get_C_indices(i_block, k_block)
                # C_idx is [c0r0, c0r1, c1r0, c1r1]
                # W index is k*n + i.
                # Local C (0,0) -> k=0, i=0 -> C_idx[0]
                # Local C (0,1) -> k=0, i=1 -> C_idx[1]
                # Local C (1,0) -> k=1, i=0 -> C_idx[2]
                # Local C (1,1) -> k=1, i=1 -> C_idx[3]
                
                # Strassen Terms
                # M1 = (A00 + A11)(B00 + B11)
                # U: A00, A11
                # V: B00, B11
                # W: C00, C11, C01, C10 with coefficients
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # Term 1: M1
                # U: A00 (+1), A11 (+1)
                # V: B00 (+1), B11 (+1)
                # W: C00 (+1), C11 (+1)
                add_term([A_idx[0], A_idx[3]], [B_idx[0], B_idx[3]], [C_idx[0], C_idx[3]], 1, 1, 1)
                
                # Term 2: M2 = (A10 + A11) B00
                # U: A10, A11
                # V: B00
                # W: C10 (+1), C11 (-1)
                add_term([A_idx[2], A_idx[3]], [B_idx[0]], [C_idx[2], C_idx[3]], 1, 1, 1)
                # Wait, W coefficients: C10 gets +1, C11 gets -1.
                # My add_term uses single sign for all W indices.
                # I need to handle mixed signs in W.
                # Let's modify add_term to take lists of (index, sign).
                
                pass # I'll rewrite the add_term logic below.
                
    # Let's refine the add_term function to handle mixed signs.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_mixed(u_list, v_list, w_list):
        # u_list: list of (idx, sign)
        # v_list: list of (idx, sign)
        # w_list: list of (idx, sign)
        
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for idx, sign in u_list:
            u_col[idx] = sign
        for idx, sign in v_list:
            v_col[idx] = sign
        for idx, sign in w_list:
            w_col[idx] = sign
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Re-run Strassen blocks
    for i_block in [1, 2]:
        for j_block in [1, 2]:
            for k_block in [1, 2]:
                A_idx = get_A_indices(i_block, j_block)
                B_idx = get_B_indices(j_block, k_block)
                C_idx = get_C_indices(i_block, k_block)
                
                # M1 = (A00 + A11)(B00 + B11)
                # C00 += M1, C11 += M1
                add_term_mixed(
                    [(A_idx[0], 1), (A_idx[3], 1)],
                    [(B_idx[0], 1), (B_idx[3], 1)],
                    [(C_idx[0], 1), (C_idx[3], 1)]
                )
                
                # M2 = (A10 + A11) B00
                # C10 += M2, C11 -= M2
                add_term_mixed(
                    [(A_idx[2], 1), (A_idx[3], 1)],
                    [(B_idx[0], 1)],
                    [(C_idx[2], 1), (C_idx[3], -1)]
                )
                
                # M3 = A00 (B10 - B11)
                # C01 += M3, C11 += M3
                add_term_mixed(
                    [(A_idx[0], 1)],
                    [(B_idx[2], 1), (B_idx[3], -1)],
                    [(C_idx[1], 1), (C_idx[3], 1)]
                )
                
                # M4 = A11 (B01 - B00)
                # C00 += M4, C10 += M4
                add_term_mixed(
                    [(A_idx[3], 1)],
                    [(B_idx[1], 1), (B_idx[0], -1)],
                    [(C_idx[0], 1), (C_idx[2], 1)]
                )
                
                # M5 = (A00 + A01) B11
                # C00 -= M5, C01 += M5
                add_term_mixed(
                    [(A_idx[0], 1), (A_idx[1], 1)],
                    [(B_idx[3], 1)],
                    [(C_idx[0], -1), (C_idx[1], 1)]
                )
                
                # M6 = (A10 - A00) (B00 + B01)
                # C11 += M6
                add_term_mixed(
                    [(A_idx[2], 1), (A_idx[0], -1)],
                    [(B_idx[0], 1), (B_idx[1], 1)],
                    [(C_idx[3], 1)]
                )
                
                # M7 = (A01 - A11) (B10 + B11)
                # C00 += M7
                add_term_mixed(
                    [(A_idx[1], 1), (A_idx[3], -1)],
                    [(B_idx[2], 1), (B_idx[3], 1)],
                    [(C_idx[0], 1)]
                )

    # 2. Remaining terms
    # Terms with j=3 or k=3
    
    # Case 1: j=3, k in {1, 2}
    # A_{i3} is 2x1. B_{3k} is 1x2.
    # A_idx: get_A_indices(i_block, 3) -> 2 indices
    # B_idx: get_B_indices(3, k_block) -> 2 indices
    # C_idx: get_C_indices(i_block, k_block) -> 4 indices
    # C_{ik} = A_{i3} B_{3k}
    # This is an outer product.
    # A_{i3} has elements A_{r, 4} for r in {0,1} (local)
    # B_{3k} has elements B_{4, c} for c in {0,1} (local)
    # C_{ik} has elements C_{c, r}
    # C_{c, r} = A_{r, 4} B_{4, c}
    # We need 4 multiplications:
    # C_{0,0} = A_{0,4} B_{4,0}
    # C_{0,1} = A_{1,4} B_{4,0}
    # C_{1,0} = A_{0,4} B_{4,1}
    # C_{1,1} = A_{1,4} B_{4,1}
    
    for i_block in [1, 2]:
        A_idx = get_A_indices(i_block, 3)
        # A_idx[0] is r=0, A_idx[1] is r=1 (local)
        
        for k_block in [1, 2]:
            B_idx = get_B_indices(3, k_block)
            # B_idx[0] is c=0, B_idx[1] is c=1 (local)
            
            C_idx = get_C_indices(i_block, k_block)
            # C_idx[0] is (c=0, r=0), C_idx[1] is (c=0, r=1)
            # C_idx[2] is (c=1, r=0), C_idx[3] is (c=1, r=1)
            
            # Term 1: C_{0,0} = A_{0,4} B_{4,0}
            add_term_mixed(
                [(A_idx[0], 1)],
                [(B_idx[0], 1)],
                [(C_idx[0], 1)]
            )
            
            # Term 2: C_{0,1} = A_{1,4} B_{4,0}
            add_term_mixed(
                [(A_idx[1], 1)],
                [(B_idx[0], 1)],
                [(C_idx[1], 1)]
            )
            
            # Term 3: C_{1,0} = A_{0,4} B_{4,1}
            add_term_mixed(
                [(A_idx[0], 1)],
                [(B_idx[1], 1)],
                [(C_idx[2], 1)]
            )
            
            # Term 4: C_{1,1} = A_{1,4} B_{4,1}
            add_term_mixed(
                [(A_idx[1], 1)],
                [(B_idx[1], 1)],
                [(C_idx[3], 1)]
            )

    # Case 2: j=3, k=3
    # A_{i3} is 2x1. B_{33} is 1x1.
    # C_{i3} is 2x1.
    # C_{r, 4} = A_{r, 4} B_{4, 4}
    
    for i_block in [1, 2]:
        A_idx = get_A_indices(i_block, 3)
        B_idx = get_B_indices(3, 3)
        C_idx = get_C_indices(i_block, 3)
        
        # Term 1: C_{0,4} = A_{0,4} B_{4,4}
        add_term_mixed(
            [(A_idx[0], 1)],
            [(B_idx[0], 1)],
            [(C_idx[0], 1)]
        )
        
        # Term 2: C_{1,4} = A_{1,4} B_{4,4}
        add_term_mixed(
            [(A_idx[1], 1)],
            [(B_idx[0], 1)],
            [(C_idx[1], 1)]
        )

    # Case 3: j in {1, 2}, k=3
    # A_{ij} is 2x2. B_{j3} is 2x1.
    # C_{i3} is 2x1.
    # C_{r, 4} = sum_{s=0}^1 A_{r, s} B_{s, 4}
    # C_{0, 4} = A_{0,0} B_{0,4} + A_{0,1} B_{1,4}
    # C_{1, 4} = A_{1,0} B_{0,4} + A_{1,1} B_{1,4}
    
    for i_block in [1, 2]:
        for j_block in [1, 2]:
            A_idx = get_A_indices(i_block, j_block)
            # A_idx[0]=A00, A_idx[1]=A01, A_idx[2]=A10, A_idx[3]=A11
            
            B_idx = get_B_indices(j_block, 3)
            # B_idx[0]=B00, B_idx[1]=B10 (local row 0,1 col 0)
            # Wait, B_{j3} is 2x1.
            # get_B_indices(j_block, 3) returns 2 indices.
            # B_idx[0] is row 0, col 0 (local)
            # B_idx[1] is row 1, col 0 (local)
            
            C_idx = get_C_indices(i_block, 3)
            # C_idx[0] is C_{0,4}, C_idx[1] is C_{1,4}
            
            # Term 1: C_{0,4} += A_{0,0} B_{0,4}
            add_term_mixed(
                [(A_idx[0], 1)],
                [(B_idx[0], 1)],
                [(C_idx[0], 1)]
            )
            
            # Term 2: C_{0,4} += A_{0,1} B_{1,4}
            add_term_mixed(
                [(A_idx[1], 1)],
                [(B_idx[1], 1)],
                [(C_idx[0], 1)]
            )
            
            # Term 3: C_{1,4} += A_{1,0} B_{0,4}
            add_term_mixed(
                [(A_idx[2], 1)],
                [(B_idx[0], 1)],
                [(C_idx[1], 1)]
            )
            
            # Term 4: C_{1,4} += A_{1,1} B_{1,4}
            add_term_mixed(
                [(A_idx[3], 1)],
                [(B_idx[1], 1)],
                [(C_idx[1], 1)]
            )

    # Convert to required format
    U = U_cols
    V = V_cols
    W = W_cols
    
    # Transpose? No, the interface expects U, V, W as lists of rows?
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # My U_cols are columns.
    # I need to return U as a list of rows.
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

