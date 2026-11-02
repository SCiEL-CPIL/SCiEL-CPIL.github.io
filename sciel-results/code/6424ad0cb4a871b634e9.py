"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by concatenating:
    # 1. Rank-26 decomposition for A * B[:, 0:3]
    # 2. Rank-26 decomposition for A * B[:, 3:6]
    # 3. Rank-9 schoolbook decomposition for A * B[:, 6:7]
    
    # Helper to add a rank-1 component
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    # But here we are dealing with sub-blocks.
    # We need to map local indices to global indices.
    
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_val, v_val, w_val, u_idx, v_idx, w_idx):
        nonlocal current_rank
        if current_rank >= rank:
            raise IndexError("Rank exceeded")
        U[u_idx][current_rank] += u_val
        V[v_idx][current_rank] += v_val
        W[w_idx][current_rank] += w_val
        current_rank += 1

    # Function to add a schoolbook decomposition for a sub-block
    # A_block: rows r0..r1, cols c0..c1 (size nr x nc)
    # B_block: rows r0..r1 (same nc), cols c0..c1 (size nc x np)
    # C_block: rows r0..r1, cols c0..c1 (size nr x np)
    def add_schoolbook(r0, c0, nr, nc, np):
        nonlocal current_rank
        for i in range(nr):
            for j in range(nc):
                for k in range(np):
                    # Global indices
                    gi = r0 + i
                    gj = c0 + j
                    gk = c0 + k # Note: B columns start at c0? 
                    # Wait, B is m x p. A is n x m.
                    # A[i, j] * B[j, k] = C[i, k]
                    # In our sub-block:
                    # A local (i, j) -> Global (r0+i, c0+j)
                    # B local (j, k) -> Global (c0+j, c0+k) ?? 
                    # No, B's rows correspond to A's columns.
                    # A's columns are 0..2. B's rows are 0..2.
                    # So if A block cols are c0..c0+nc-1, B block rows are c0..c0+nc-1.
                    # B block cols are k0..k0+np-1.
                    
                    # Let's redefine parameters to be clearer.
                    # a_r0, a_c0, nr, nc
                    # b_r0 (must match a_c0), b_c0, nc, np
                    # c_r0 (must match a_r0), c_c0 (must match b_c0), nr, np
                    
                    u_idx = (r0 + i) * m + (c0 + j)
                    v_idx = (c0 + j) * p + (c0 + k) # WRONG. B col index is global col in B.
                    
                    # Let's pass offsets explicitly.
                    pass

    # Let's refactor add_schoolbook to take global offsets.
    def add_schoolbook_block(a_r0, a_c0, b_c0, nr, nc, np):
        nonlocal current_rank
        for i in range(nr):
            for j in range(nc):
                for k in range(np):
                    gi = a_r0 + i
                    gj = a_c0 + j
                    gk = b_c0 + k
                    
                    u_idx = gi * m + gj
                    v_idx = gj * p + gk
                    w_idx = gk * n + gi
                    
                    U[u_idx][current_rank] = 1
                    V[v_idx][current_rank] = 1
                    W[w_idx][current_rank] = 1
                    current_rank += 1

    # Function to add Strassen decomposition for 2x2 block
    def add_strassen_2x2(a_r0, a_c0, b_c0):
        # A block: rows a_r0, a_r0+1; cols a_c0, a_c0+1
        # B block: rows a_c0, a_c0+1; cols b_c0, b_c0+1
        # C block: rows a_r0, a_r0+1; cols b_c0, b_c0+1
        
        # Local indices:
        # A: a=A00, b=A01, c=A10, d=A11
        # B: e=B00, f=B01, g=B10, h=B11
        
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(g-h)
        # m4 = d(f-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+g)
        # m7 = (b-d)(f+h)
        
        # We need to add these as linear combinations in U, V, W.
        # Each mk is a product of a linear form in A and a linear form in B.
        # So we add 7 terms.
        
        def add_prod(au, av, bu, bv):
            # au, av are coefficients for A in U
            # bu, bv are coefficients for B in V
            # We need to map local (i,j) to global.
            nonlocal current_rank
            
            # U part: sum of au[i][j] * A_local[i][j]
            # This corresponds to adding to U rows.
            # For each non-zero coeff in au, we add coeff to U[global_idx][current_rank]
            
            # But we are adding a single rank-1 term: (u_vec) * (v_vec) * (w_vec)
            # Here u_vec is the vector of coefficients for A.
            # v_vec is the vector of coefficients for B.
            # w_vec is the vector of coefficients for C.
            
            # Wait, the standard decomposition is:
            # C = sum_k U[:, k] * V[:, k] * W[:, k]^T
            # So for each multiplication mk, we define:
            # u_k: vector of size n*m (only non-zero in the 2x2 block)
            # v_k: vector of size m*p (only non-zero in the 2x2 block)
            # w_k: vector of size p*n (only non-zero in the 2x2 block)
            
            pass

        # Let's implement the 7 terms explicitly.
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # U coeffs: A00:1, A11:1
        # V coeffs: B00:1, B11:1
        # W coeffs: C00:1, C01:-1, C10:-1, C11:1  (from c00=m1..., c11=m1...)
        # c00 = m1 + m4 - m5 + m7 -> coeff 1
        # c01 = m3 + m5 -> coeff 0
        # c10 = m2 + m4 -> coeff 0
        # c11 = m1 - m2 + m3 + m6 -> coeff 1
        # Wait, c01, c10 don't have m1?
        # c01 = m3 + m5. No m1.
        # c10 = m2 + m4. No m1.
        # So W coeffs: C00:1, C11:1.
        
        # Let's verify c00 and c11 again.
        # c00 = m1 + m4 - m5 + m7
        # c11 = m1 - m2 + m3 + m6
        # Yes.
        
        # We need a helper to add a term with specific U, V, W coefficients.
        
        def add_rank1(u_coeffs, v_coeffs, w_coeffs):
            nonlocal current_rank
            # u_coeffs: dict {(i,j): val}
            # v_coeffs: dict {(j,k): val}
            # w_coeffs: dict {(k,i): val}
            
            for (i, j), val in u_coeffs.items():
                if val != 0:
                    U[a_r0 + i, a_c0 + j][current_rank] += val
            for (j, k), val in v_coeffs.items():
                if val != 0:
                    V[a_c0 + j, b_c0 + k][current_rank] += val
            for (k, i), val in w_coeffs.items():
                if val != 0:
                    W[b_c0 + k, a_r0 + i][current_rank] += val
            current_rank += 1

        # 1. m1 = (A00+A11)(B00+B11)
        add_rank1(
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        )
        
        # 2. m2 = (A10+A11)B00
        # c10 = m2 + m4 -> coeff 1
        # c11 = m1 - m2 + ... -> coeff -1
        add_rank1(
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1} # W[k,i] -> k=0,i=1 is C10? No.
            # W index is k*n + i.
            # C10 is i=1, k=0.
            # C11 is i=1, k=1.
            # So W[0*3+1]=1, W[1*3+1]=-1.
            # My dict keys are (k,i).
            # (0,1): 1, (1,1): -1.
        )
        
        # 3. m3 = A00(B10-B11)
        # c01 = m3 + m5 -> coeff 1
        # c11 = m1 - m2 + m3 + m6 -> coeff 1
        add_rank1(
            {(0,0):1},
            {(1,0):1, (1,1):-1},
            {(1,0):1, (1,1):1} # C01 (k=1,i=0), C11 (k=1,i=1)
        )
        
        # 4. m4 = A11(B01-B00)
        # c00 = m1 + m4 - ... -> coeff 1
        # c10 = m2 + m4 -> coeff 1
        add_rank1(
            {(1,1):1},
            {(0,1):1, (0,0):-1},
            {(0,0):1, (0,1):1} # C00 (k=0,i=0), C10 (k=0,i=1)
        )
        
        # 5. m5 = (A00+A01)B11
        # c00 = ... - m5 ... -> coeff -1
        # c01 = m3 + m5 -> coeff 1
        add_rank1(
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,0):-1, (1,0):1} # C00 (k=0,i=0), C01 (k=1,i=0)
        )
        
        # 6. m6 = (A10-A00)(B00+B10)
        # c11 = ... + m6 -> coeff 1
        add_rank1(
            {(1,0):1, (0,0):-1},
            {(0,0):1, (1,0):1},
            {(1,1):1} # C11 (k=1,i=1)
        )
        
        # 7. m7 = (A01-A11)(B11+B01)
        # c00 = ... + m7 -> coeff 1
        add_rank1(
            {(0,1):1, (1,1):-1},
            {(1,1):1, (0,1):1},
            {(0,0):1} # C00 (k=0,i=0)
        )

    # Apply to Block 1 (Cols 0-2 of B)
    # A is 3x3.
    # Strassen on A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0)
    
    # Remaining terms for Block 1 (Schoolbook)
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (2x1 * 1x2)
    add_schoolbook_block(0, 2, 0, 2, 1, 2) # a_r0=0, a_c0=2, b_c0=0, nr=2, nc=1, np=2
    
    # A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2] (2x2 * 2x1)
    add_schoolbook_block(0, 0, 2, 2, 2, 1)
    
    # A[0:2, 2] * B[2, 2] -> C[0:2, 2] (2x1 * 1x1)
    add_schoolbook_block(0, 2, 2, 2, 1, 1)
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2 * 2x2)
    add_schoolbook_block(2, 0, 0, 1, 2, 2)
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2] (1x1 * 1x2)
    add_schoolbook_block(2, 2, 0, 1, 1, 2)
    
    # A[2, 0:2] * B[0:2, 2] -> C[2, 2] (1x2 * 2x1)
    add_schoolbook_block(2, 0, 2, 1, 2, 1)
    
    # A[2, 2] * B[2, 2] -> C[2, 2] (1x1 * 1x1)
    add_schoolbook_block(2, 2, 2, 1, 1, 1)

    # Apply to Block 2 (Cols 3-5 of B)
    # Shift B cols by 3.
    add_strassen_2x2(0, 0, 3)
    add_schoolbook_block(0, 2, 3, 2, 1, 2)
    add_schoolbook_block(0, 0, 5, 2, 2, 1)
    add_schoolbook_block(0, 2, 5, 2, 1, 1)
    add_schoolbook_block(2, 0, 3, 1, 2, 2)
    add_schoolbook_block(2, 2, 3, 1, 1, 2)
    add_schoolbook_block(2, 0, 5, 1, 2, 1)
    add_schoolbook_block(2, 2, 5, 1, 1, 1)

    # Apply to Block 3 (Col 6 of B)
    # Schoolbook 3x3 * 3x1
    add_schoolbook_block(0, 0, 6, 3, 3, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
