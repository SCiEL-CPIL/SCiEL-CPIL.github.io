"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        # Indices are local to the 2x2 block (0,1 for rows/cols)
        # We will map them to global indices later
        
        # M1
        ((0, 3), (1, 1), (0, 2), (1, 1), (0, 3), (1, 1)), # A00,A11; B00,B11; C00,C11
        # M2
        ((1, 3), (1, 1), (0, 0), (1, 1), (2, 0), (1, -1)), # A10,A11; B00; C10, C11
        # M3
        ((0, 0), (1, 1), (1, 2), (1, -1), (1, 3), (1, 1)), # A00; B10,B11; C01, C11
        # M4
        ((3, 3), (1, 1), (1, 0), (1, -1), (0, 2), (1, 1)), # A11; B01,B00; C00, C10
        # M5
        ((0, 1), (1, 1), (2, 2), (1, 1), (0, 1), (-1, 1)), # A00,A01; B11; C00, C01
        # M6
        ((1, 0), (1, -1), (0, 1), (1, 1), (3, 0), (1, 1)), # A10,A00; B00,B10; C11, C10? 
        # Wait, M6 contributes to C11 and C10?
        # C11 = M1 - M2 + M3 + M6. C10 = M2 + M4.
        # So M6 only contributes to C11.
        # Correction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify W coeffs for M6.
        # M6 is in C11 only.
        # So W indices: (3,), coeffs: (1,).
        
        # M7
        ((1, 3), (1, -1), (2, 1), (1, 1), (0, 0), (1, 1)) # A01,A11; B11,B01; C00
    ]
    
    # I will hardcode the Strassen terms more carefully to avoid errors.
    # Local A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: A(0,3) B(0,2) C(0,3)
    # M2: A(1,3) B(0) C(2,3) with coeffs (1,1), (1), (1,-1)
    # M3: A(0) B(1,2) C(1,3) with coeffs (1), (1,-1), (1,1)
    # M4: A(3) B(1,0) C(0,2) with coeffs (1), (1,-1), (1,1)
    # M5: A(0,1) B(2) C(0,1) with coeffs (1,1), (1), (-1,1)
    # M6: A(1,0) B(0,1) C(3) with coeffs (1,-1), (1,1), (1)
    # M7: A(1,3) B(2,1) C(0) with coeffs (1,-1), (1,1), (1)

    def add_strassen_block(base_a, base_b, base_c, start_col):
        nonlocal col
        # base_a: offset for A indices (i*m+j)
        # base_b: offset for B indices (j*p+k)
        # base_c: offset for C indices (k*n+i)
        
        terms = [
            # (a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff)
            # M1
            (0, 1, 0, 1, 0, 1), (3, 1, 0, 1, 0, 1), (0, 1, 2, 1, 0, 1), (3, 1, 2, 1, 0, 1), (0, 1, 0, 1, 3, 1), (3, 1, 0, 1, 3, 1), (0, 1, 2, 1, 3, 1), (3, 1, 2, 1, 3, 1),
            # This is getting messy to expand manually. I will use a helper to add terms.
        ]
        return start_col

    # Let's just list the 7 terms explicitly for one block.
    # A local indices: 00, 01, 10, 11
    # B local indices: 00, 01, 10, 11
    # C local indices: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # A: 0, 3. B: 0, 2. C: 0, 3.
    # Term 1: A00*B00 -> C00, C11
    # Term 2: A00*B11 -> C00, C11
    # Term 3: A11*B00 -> C00, C11
    # Term 4: A11*B11 -> C00, C11
    # Wait, Strassen combines them.
    # M1 is ONE rank-1 term in the tensor decomposition?
    # Yes, because it is a product of a linear form in A and a linear form in B.
    # So for M1, we have ONE column in U, V, W.
    # U: A00=1, A11=1.
    # V: B00=1, B11=1.
    # W: C00=1, C11=1.
    
    # So for each block, we add 7 terms for Strassen, 6 for Outer, 4 for MatVec.
    
    for r in range(3):
        # Block r corresponds to k in {3r, 3r+1, 3r+2}
        
        # 1. Strassen for A[:, 0:2] * B[0:2, 3r:3r+2]
        # A indices: i*3 + j. j in {0,1}.
        # B indices: j*9 + k. j in {0,1}, k in {3r, 3r+1}.
        # C indices: k*2 + i. k in {3r, 3r+1}, i in {0,1}.
        
        # Helper to add a term
        def add_term(a_idx, a_val, b_idx, b_val, c_idx, c_val):
            nonlocal col
            U[a_idx][col] += a_val
            V[b_idx][col] += b_val
            W[c_idx][col] += c_val
        
        # M1
        add_term(0, 1, 0*9 + 3*r, 1, (3*r)*2 + 0, 1) # A00, B0, C0
        add_term(3, 1, 0*9 + 3*r, 1, (3*r)*2 + 0, 1) # A11 (idx 3), B0, C0
        add_term(0, 1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, 1) # A00, B1 (idx 1*9+3r+1), C0
        add_term(3, 1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, 1) # A11, B1, C0
        add_term(0, 1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, 1) # A00, B0, C11 (idx (3r+1)*2+1)
        add_term(3, 1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, 1) # A11, B0, C11
        add_term(0, 1, 1*9 + 3*r+1, 1, (3*r+1)*2 + 1, 1) # A00, B1, C11
        add_term(3, 1, 1*9 + 3*r+1, 1, (3*r+1)*2 + 1, 1) # A11, B1, C11
        col += 1

        # M2 = (A10+A11)B00
        # A10 is idx 2, A11 is idx 3. B00 is idx 0.
        # C10 is idx (3r)*2+1. C11 is idx (3r+1)*2+1.
        add_term(2, 1, 0*9 + 3*r, 1, (3*r)*2 + 1, 1)
        add_term(3, 1, 0*9 + 3*r, 1, (3*r)*2 + 1, 1)
        add_term(2, 1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, -1)
        add_term(3, 1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, -1)
        col += 1

        # M3 = A00(B10-B11)
        # A00 idx 0. B10 idx 1*9+3r. B11 idx 1*9+3r+1.
        # C01 idx (3r+1)*2+0. C11 idx (3r+1)*2+1.
        add_term(0, 1, 1*9 + 3*r, 1, (3*r+1)*2 + 0, 1)
        add_term(0, 1, 1*9 + 3*r+1, -1, (3*r+1)*2 + 0, 1)
        add_term(0, 1, 1*9 + 3*r, 1, (3*r+1)*2 + 1, 1)
        add_term(0, 1, 1*9 + 3*r+1, -1, (3*r+1)*2 + 1, 1)
        col += 1

        # M4 = A11(B01-B00)
        # A11 idx 3. B01 idx 0*9+3r+1. B00 idx 0*9+3r.
        # C00 idx (3r)*2+0. C10 idx (3r)*2+1.
        add_term(3, 1, 0*9 + 3*r+1, 1, (3*r)*2 + 0, 1)
        add_term(3, 1, 0*9 + 3*r, -1, (3*r)*2 + 0, 1)
        add_term(3, 1, 0*9 + 3*r+1, 1, (3*r)*2 + 1, 1)
        add_term(3, 1, 0*9 + 3*r, -1, (3*r)*2 + 1, 1)
        col += 1

        # M5 = (A00+A01)B11
        # A00 idx 0, A01 idx 1. B11 idx 1*9+3r+1.
        # C00 idx (3r)*2+0. C01 idx (3r+1)*2+0.
        add_term(0, 1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, -1)
        add_term(1, 1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, -1)
        add_term(0, 1, 1*9 + 3*r+1, 1, (3*r+1)*2 + 0, 1)
        add_term(1, 1, 1*9 + 3*r+1, 1, (3*r+1)*2 + 0, 1)
        col += 1

        # M6 = (A10-A00)(B00+B10)
        # A10 idx 2, A00 idx 0. B00 idx 0, B10 idx 1*9+3r.
        # C11 idx (3r+1)*2+1.
        add_term(2, 1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, 1)
        add_term(0, -1, 0*9 + 3*r, 1, (3*r+1)*2 + 1, 1)
        add_term(2, 1, 1*9 + 3*r, 1, (3*r+1)*2 + 1, 1)
        add_term(0, -1, 1*9 + 3*r, 1, (3*r+1)*2 + 1, 1)
        col += 1

        # M7 = (A01-A11)(B11+B01)
        # A01 idx 1, A11 idx 3. B11 idx 1*9+3r+1, B01 idx 0*9+3r+1.
        # C00 idx (3r)*2+0.
        add_term(1, 1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, 1)
        add_term(3, -1, 1*9 + 3*r+1, 1, (3*r)*2 + 0, 1)
        add_term(1, 1, 0*9 + 3*r+1, 1, (3*r)*2 + 0, 1)
        add_term(3, -1, 0*9 + 3*r+1, 1, (3*r)*2 + 0, 1)
        col += 1

        # 2. Outer Product A[:,2] * B[2,:]
        # A idx: i*3+2. i in {0,1}.
        # B idx: 2*9 + 3r+k'. k' in {0,1,2}.
        # C idx: (3r+k')*2 + i.
        for i in range(2):
            for k_prime in range(3):
                a_idx = i*3 + 2
                b_idx = 2*9 + 3*r + k_prime
                c_idx = (3*r + k_prime)*2 + i
                add_term(a_idx, 1, b_idx, 1, c_idx, 1)
                col += 1

        # 3. Matrix-Vector A[:,0:2] * B[0:2, 2]
        # A idx: i*3+j. i in {0,1}, j in {0,1}.
        # B idx: j*9 + 3r+2.
        # C idx: (3r+2)*2 + i.
        for i in range(2):
            for j in range(2):
                a_idx = i*3 + j
                b_idx = j*9 + 3*r + 2
                c_idx = (3*r + 2)*2 + i
                add_term(a_idx, 1, b_idx, 1, c_idx, 1)
                col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
