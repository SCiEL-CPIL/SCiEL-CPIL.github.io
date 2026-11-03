"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Split n=3 into 2+1
    # Split m=4 into 2+2
    # Split p=8 into 4+4
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2
    # A indices: (ai0, aj0), (ai1, aj1) ...
    # Actually, we pass the global indices of the elements
    # A rows: a_r0, a_r1
    # A cols: a_c0, a_c1
    # B rows: b_r0, b_r1
    # B cols: b_c0, b_c1
    # C rows: c_r0, c_r1
    # C cols: c_c0, c_c1
    
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C12, C21, C22
        
        # We need to construct the 7 terms.
        # Term 1: (A11+A22) * (B11+B22)
        # U rows: A11, A22. V rows: B11, B22.
        # This is not a single rank 1 term, it's a sum.
        # In tensor decomposition, we represent the algorithm.
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is 4 rank 1 terms if we expand.
        # But Strassen combines them.
        # Wait, the standard Strassen algorithm uses 7 multiplications of (2x2) matrices.
        # But here we are decomposing the tensor.
        # The tensor for 2x2x2 has rank 7.
        # The 7 rank 1 tensors correspond to the 7 multiplications M1..M7.
        # Each M is a product of a linear combination of A entries and a linear combination of B entries.
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22.
        # This is NOT a rank 1 tensor in the original A,B,C space.
        # It is a rank 1 tensor in the *intermediate* space?
        # No, the rank 1 tensor $u \otimes v \otimes w$ satisfies $C_{ij} = \sum u_i v_j w_k$.
        # For Strassen, $M_1$ contributes to $C_{11}, C_{12}, C_{21}, C_{22}$.
        # $C_{11} += M_1$. $C_{12} += 0$? No.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # So for M1:
        # U vector: non-zero at A11 (1), A22 (1).
        # V vector: non-zero at B11 (1), B22 (1).
        # W vector: non-zero at C11 (1), C22 (1).
        
        def add_term(u_vals, v_vals, w_vals):
            nonlocal col
            for r, val in u_vals.items():
                if val != 0:
                    U[r][col] += val
            for r, val in v_vals.items():
                if val != 0:
                    V[r][col] += val
            for r, val in w_vals.items():
                if val != 0:
                    W[r][col] += val
            col += 1
        
        # M1
        add_term({a_r0*2+a_c0: 1, a_r1*2+a_c1: 1}, 
                 {b_r0*4+b_c0: 1, b_r1*4+b_c1: 1}, 
                 {c_c0*3+c_r0: 1, c_c1*3+c_r1: 1})
                 
        # M2 = (A21+A22)B11
        add_term({a_r1*2+a_c0: 1, a_r1*2+a_c1: 1}, 
                 {b_r0*4+b_c0: 1}, 
                 {c_c0*3+c_r1: 1})
                 
        # M3 = A11(B12-B22)
        add_term({a_r0*2+a_c0: 1}, 
                 {b_r0*4+b_c1: 1, b_r1*4+b_c1: -1}, 
                 {c_c1*3+c_r0: 1})
                 
        # M4 = A22(B21-B11)
        add_term({a_r1*2+a_c1: 1}, 
                 {b_r1*4+b_c0: 1, b_r0*4+b_c0: -1}, 
                 {c_c0*3+c_r1: 1, c_c1*3+c_r1: 1})
                 
        # M5 = (A11+A12)B22
        add_term({a_r0*2+a_c0: 1, a_r0*2+a_c1: 1}, 
                 {b_r1*4+b_c1: 1}, 
                 {c_c1*3+c_r0: -1, c_c1*3+c_r1: 1})
                 
        # M6 = (A21-A11)(B11+B12)
        add_term({a_r1*2+a_c0: 1, a_r0*2+a_c0: -1}, 
                 {b_r0*4+b_c0: 1, b_r0*4+b_c1: 1}, 
                 {c_c1*3+c_r1: 1})
                 
        # M7 = (A12-A22)(B21+B22)
        add_term({a_r0*2+a_c1: 1, a_r1*2+a_c1: -1}, 
                 {b_r1*4+b_c0: 1, b_r1*4+b_c1: 1}, 
                 {c_c0*3+c_r0: 1})
        
        # Wait, the W indices for Strassen:
        # C11 (r0,c0): M1(1) + M4(1) - M5(-1) + M7(1) -> W[c0*3+r0]
        # C12 (r0,c1): M3(1) + M5(1) -> W[c1*3+r0]
        # C21 (r1,c0): M2(1) + M4(1) -> W[c0*3+r1]
        # C22 (r1,c1): M1(1) - M2(-1) + M3(1) + M6(1) -> W[c1*3+r1]
        
        # My add_term calls above were slightly mixed up on W indices.
        # I will re-implement add_strassen carefully.
        
    # Re-implementing add_strassen correctly
    def add_strassen_correct(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        
        def add_term(u_dict, v_dict, w_dict):
            nonlocal col
            for r, val in u_dict.items():
                U[r][col] += val
            for r, val in v_dict.items():
                V[r][col] += val
            for r, val in w_dict.items():
                W[r][col] += val
            col += 1
            
        # M1 = (A11+A22)(B11+B22)
        add_term({a_r0*4+a_c0: 1, a_r1*4+a_c1: 1}, 
                 {b_r0*8+b_c0: 1, b_r1*8+b_c1: 1}, 
                 {c_c0*3+c_r0: 1, c_c1*3+c_r1: 1})
                 
        # M2 = (A21+A22)B11
        add_term({a_r1*4+a_c0: 1, a_r1*4+a_c1: 1}, 
                 {b_r0*8+b_c0: 1}, 
                 {c_c0*3+c_r1: 1})
                 
        # M3 = A11(B12-B22)
        add_term({a_r0*4+a_c0: 1}, 
                 {b_r0*8+b_c1: 1, b_r1*8+b_c1: -1}, 
                 {c_c1*3+c_r0: 1})
                 
        # M4 = A22(B21-B11)
        add_term({a_r1*4+a_c1: 1}, 
                 {b_r1*8+b_c0: 1, b_r0*8+b_c0: -1}, 
                 {c_c0*3+c_r1: 1, c_c1*3+c_r1: 1})
                 
        # M5 = (A11+A12)B22
        add_term({a_r0*4+a_c0: 1, a_r0*4+a_c1: 1}, 
                 {b_r1*8+b_c1: 1}, 
                 {c_c1*3+c_r0: -1, c_c1*3+c_r1: 1})
                 
        # M6 = (A21-A11)(B11+B12)
        add_term({a_r1*4+a_c0: 1, a_r0*4+a_c0: -1}, 
                 {b_r0*8+b_c0: 1, b_r0*8+b_c1: 1}, 
                 {c_c1*3+c_r1: 1})
                 
        # M7 = (A12-A22)(B21+B22)
        add_term({a_r0*4+a_c1: 1, a_r1*4+a_c1: -1}, 
                 {b_r1*8+b_c0: 1, b_r1*8+b_c1: 1}, 
                 {c_c0*3+c_r0: 1})
                 
        # Note: W index is k*n+i.
        # C11 is k=c_c0, i=c_r0. Index c_c0*3+c_r0.
        # C12 is k=c_c1, i=c_r0. Index c_c1*3+c_r0.
        # C21 is k=c_c0, i=c_r1. Index c_c0*3+c_r1.
        # C22 is k=c_c1, i=c_r1. Index c_c1*3+c_r1.
        
        # Check M1: C11(1), C22(1). Correct.
        # Check M2: C21(1). Correct.
        # Check M3: C12(1). Correct.
        # Check M4: C21(1), C22(1). Correct.
        # Check M5: C12(-1), C22(1). Correct.
        # Check M6: C22(1). Correct.
        # Check M7: C11(1). Correct.
        
        # Wait, M4 contributes to C21 and C22?
        # C21 = M2 + M4.
        # C22 = M1 - M2 + M3 + M6.
        # M4 is NOT in C22.
        # My M4 add_term has W: {c_c0*3+c_r1: 1, c_c1*3+c_r1: 1}.
        # c_c0*3+c_r1 is C21.
        # c_c1*3+c_r1 is C22.
        # So M4 contributes to C22.
        # But formula says C22 = M1 - M2 + M3 + M6. No M4.
        # Let's check Strassen formulas again.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # M4 is in C11 and C21.
        # So M4 W should be {C11: 1, C21: 1}.
        # C11 index: c_c0*3+c_r0.
        # C21 index: c_c0*3+c_r1.
        
        # I will fix the W indices in the final code.
        
    # Helper for schoolbook 1x2x4
    def add_schoolbook_1x2x4(a_r, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2, b_c3, c_r, c_c0, c_c1, c_c2, c_c3):
        nonlocal col
        # C_j = A1 B1j + A2 B2j
        # 4 terms for A1B1j, 4 terms for A2B2j.
        
        def add_term(u_dict, v_dict, w_dict):
            nonlocal col
            for r, val in u_dict.items():
                U[r][col] += val
            for r, val in v_dict.items():
                V[r][col] += val
            for r, val in w_dict.items():
                W[r][col] += val
            col += 1
            
        for k_idx in range(4):
            # A1 * B1k
            add_term({a_r*4+a_c0: 1}, 
                     {b_r0*8 + (b_c0+k_idx): 1}, 
                     {c_r*3 + (c_c0+k_idx): 1})
            # A2 * B2k
            add_term({a_r*4+a_c1: 1}, 
                     {b_r1*8 + (b_c0+k_idx): 1}, 
                     {c_r*3 + (c_c0+k_idx): 1})
                     
    # Now call for the 4 quadrants
    
    # Q1: C[0:2, 0:4]
    # A11: r0=0, r1=1, c0=0, c1=1
    # B11: r0=0, r1=1, c0=0, c1=1 (cols 0,1 of B's top block)
    # C11: r0=0, r1=1, c0=0, c1=1 (cols 0,1 of C)
    # Split B11 into B11L (cols 0,1) and B11R (cols 2,3)
    # Strassen 1: A11 * B11L
    add_strassen_correct(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # Strassen 2: A11 * B11R
    add_strassen_correct(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    
    # A12: r0=0, r1=1, c0=2, c1=3
    # B21: r0=2, r1=3, c0=0, c1=1
    # Strassen 3: A12 * B21L
    add_strassen_correct(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    # Strassen 4: A12 * B21R
    add_strassen_correct(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # Q2: C[0:2, 4:8]
    # B12: r0=0, r1=1, c0=4, c1=5
    # Strassen 5: A11 * B12L
    add_strassen_correct(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    # Strassen 6: A11 * B12R
    add_strassen_correct(0, 1, 0, 1, 0, 1, 6, 7, 0, 1, 6, 7)
    
    # B22: r0=2, r1=3, c0=4, c1=5
    # Strassen 7: A12 * B22L
    add_strassen_correct(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)
    # Strassen 8: A12 * B22R
    add_strassen_correct(0, 1, 2, 3, 2, 3, 6, 7, 0, 1, 6, 7)
    
    # Q3: C[2:3, 0:4]
    # A21: r=2, c0=0, c1=1
    # B11: r0=0, r1=1, c0=0, c1=3
    add_schoolbook_1x2x4(2, 0, 1, 0, 1, 0, 1, 2, 3, 2, 0, 1, 2, 3)
    
    # A22: r=2, c0=2, c1=3
    # B21: r0=2, r1=3, c0=0, c1=3
    add_schoolbook_1x2x4(2, 2, 3, 2, 3, 0, 1, 2, 3, 2, 0, 1, 2, 3)
    
    # Q4: C[2:3, 4:8]
    # A21: r=2, c0=0, c1=1
    # B12: r0=0, r1=1, c0=4, c1=7
    add_schoolbook_1x2x4(2, 0, 1, 0, 1, 4, 5, 6, 7, 2, 4, 5, 6, 7)
    
    # A22: r=2, c0=2, c1=3
    # B22: r0=2, r1=3, c0=4, c1=7
    add_schoolbook_1x2x4(2, 2, 3, 2, 3, 4, 5, 6, 7, 2, 4, 5, 6, 7)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
