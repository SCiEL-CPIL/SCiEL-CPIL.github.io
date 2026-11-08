import time

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - C1 = A * B1 using recursive Strassen (rank 49).
    - C2 = A * B2 using recursive Strassen (rank 49).
    - C3 = A * B3 using schoolbook (rank 16).
    - Total rank: 114.
    
    Interface:
    U: (16, rank) where U[i*4+j, r] is coeff for A[i,j]
    V: (36, rank) where V[j*9+k, r] is coeff for B[j,k]
    W: (36, rank) where W[k*4+i, r] is coeff for C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = 4, 4, 9
    # U dim: n*m = 16
    # V dim: m*p = 36
    # W dim: p*n = 36
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2x2: 7 terms
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # Keys are (row, col) in 0..1
    s2_terms = [
        # 1. (A00+A11)(B00+B11) -> C00 +, C11 +
        (({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})),
        # 2. (A10+A11)B00 -> C10 +, C11 -
        (({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})),
        # 3. A00(B01-B11) -> C01 +, C11 +
        (({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})),
        # 4. A11(B10-B00) -> C00 +, C10 +
        (({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})),
        # 5. (A00+A01)B11 -> C00 -, C01 +
        (({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})),
        # 6. (A10-A00)(B00+B01) -> C11 +
        (({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})),
        # 7. (A01-A11)(B10+B11) -> C00 +
        (({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})),
    ]

    def add_strassen_4x4(b_col_off):
        """
        Add 49 terms for A(4x4) * B_block(4x4) -> C_block(4x4).
        b_col_off: starting column index in B (0 or 4).
        C columns will be at the same offset.
        """
        for oa, ob, oc in s2_terms:
            for ia, ib, ic in s2_terms:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Outer indices: oor, ooc in {0,1}
                # Inner indices: ior, ioc in {0,1}
                
                # U vector: A indices
                # A global row: oor*2 + ior
                # A global col: ooc*2 + ioc
                for (oor, ooc), c_ao in oa.items():
                    for (ior, ioc), c_ia in ia.items():
                        r = oor * 2 + ior
                        c = ooc * 2 + ioc
                        u[r * 4 + c] += c_ao * c_ia
                
                # V vector: B indices
                # B global row: oor*2 + ior
                # B global col: b_col_off + ooc*2 + ioc
                for (oor, ooc), c_bo in ob.items():
                    for (ior, ioc), c_ib in ib.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        v[r * 9 + c] += c_bo * c_ib
                
                # W vector: C indices
                # C global row: oor*2 + ior
                # C global col: b_col_off + ooc*2 + ioc
                # Note: oc determines which C block gets the contribution
                # In Strassen, oc is a dict {(r,c): coeff}
                # The contribution goes to C block (oor, ooc) ? 
                # No, the outer Strassen term maps A-blocks and B-blocks to C-blocks.
                # Let's re-verify the Strassen mapping.
                
                # Standard Strassen:
                # M1 = (A00+A11)(B00+B11)
                # C00 = M1 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M3 + M6
                
                # My s2_terms list encodes:
                # Term 1: A: {(0,0):1, (1,1):1}, B: {(0,0):1, (1,1):1}, C: {(0,0):1, (1,1):1}
                # This means:
                # The product of (A00+A11) and (B00+B11) contributes +1 to C00 and +1 to C11.
                # This matches M1.
                
                # So, for a given outer term (oa, ob, oc):
                # The "result" is a 2x2 matrix of blocks.
                # The entry (oor, ooc) in this result matrix has coefficient c_co from oc.
                # This result block is then multiplied? No.
                # In the recursive Strassen, the "multiplication" of the blocks IS the inner Strassen.
                
                # So, the inner Strassen decomposes the 2x2x2 multiplication of the block values.
                # The inner terms (ia, ib, ic) define the rank-1 decomposition of the 2x2x2 tensor.
                
                # How do they combine?
                # The outer term specifies WHICH blocks of A and B are involved.
                # The inner term specifies HOW the 2x2 elements within those blocks are combined.
                
                # Wait, the recursive Strassen for 4x4x4:
                # We view A as 2x2 blocks of 2x2 matrices.
                # A_block(r, c) is a 2x2 matrix.
                # The outer Strassen term combines A_blocks and B_blocks.
                # e.g., M1 = (A00 + A11) * (B00 + B11).
                # Here, (A00+A11) is a 2x2 matrix (sum of two 2x2 matrices).
                # (B00+B11) is a 2x2 matrix.
                # The product is a 2x2 matrix.
                # This product contributes to C00 and C11 (coeff 1 each).
                
                # So, the term in the 4x4x4 tensor is:
                # Sum over inner elements of (A00+A11) and (B00+B11) to form C00 and C11.
                
                # My current loop:
                # for (oor, ooc) in oa:  # oor, ooc are block indices (0 or 1)
                #    for (ior, ioc) in ia: # ior, ioc are element indices within block (0 or 1)
                
                # This correctly constructs the vector for A:
                # u[global_r, global_c] += c_ao * c_ia
                # This represents the linear combination of A elements.
                
                # Similarly for V and W.
                
                # BUT, the W vector construction needs to map the inner C block to the correct global position.
                # The outer term `oc` tells us which C BLOCKS receive the contribution.
                # e.g., if oc has {(0,0):1}, it means the result contributes to C block (0,0).
                # The inner term `ic` tells us which ELEMENTS within that C block are affected.
                
                # So:
                # for (oor, ooc), c_co in oc.items():  # oor, ooc are C block indices
                #    for (ior, ioc), c_ic in ic.items(): # ior, ioc are element indices within C block
                #       global_r = oor * 2 + ior
                #       global_c = b_col_off + ooc * 2 + ioc
                #       w[global_c * 4 + global_r] += c_co * c_ic
                
                for (oor, ooc), c_co in oc.items():
                    for (ior, ioc), c_ic in ic.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        w[c * 4 + r] += c_co * c_ic

                add_term(u, v, w)

    def add_schoolbook_4x1(b_col):
        """
        Add 16 terms for A(4x4) * B_col(4x1) -> C_col(4x1).
        b_col: column index in B (8).
        """
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                u[i * 4 + j] = 1.0
                v[j * 9 + b_col] = 1.0
                # C[i, b_col]
                w[b_col * 4 + i] = 1.0
                
                add_term(u, v, w)

    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook 144 if somehow we run out of time (should not happen)
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0] * 16
                    v = [0.0] * 36
                    w = [0.0] * 36
                    u[i * 4 + j] = 1.0
                    v[j * 9 + k] = 1.0
                    w[k * 4 + i] = 1.0
                    add_term(u, v, w)
    else:
        # Add Strassen for B cols 0-3
        add_strassen_4x4(0)
        
        # Add Strassen for B cols 4-7
        add_strassen_4x4(4)
        
        # Add Schoolbook for B col 8
        add_schoolbook_4x1(8)

    rank = len(U_cols)
    
    # Build output matrices
    # U: 16 x rank
    U = [[U_cols[r][i] for r in range(rank)] for i in range(16)]
    # V: 36 x rank
    V = [[V_cols[r][i] for r in range(rank)] for i in range(36)]
    # W: 36 x rank
    W = [[W_cols[r][i] for r in range(rank)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
