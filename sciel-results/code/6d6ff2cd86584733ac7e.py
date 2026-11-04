def construct():
    n, m, p = 4, 5, 6
    # Indices for U: i*m + j (i in 0..3, j in 0..4) -> 0..19
    # Indices for V: j*p + k (j in 0..4, k in 0..5) -> 0..29
    # Indices for W: k*n + i (k in 0..5, i in 0..3) -> 0..23
    
    # We will store terms as lists of (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w, coeff=1):
        terms.append((u, v, w, coeff))

    # --- Part 1: A_R (4x1) * B_B (1x6) ---
    # A_R is A[:, 4] (col 4)
    # B_B is B[4, :] (row 4)
    # C[i, k] += A[i, 4] * B[4, k]
    for i in range(4):
        for k in range(6):
            u_idx = i * 5 + 4
            v_idx = 4 * 6 + k
            w_idx = k * 4 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # --- Part 2: A_L (4x4) * B_T (4x6) ---
    # A_L is A[:, 0:4]
    # B_T is B[0:4, :]
    
    # Split B_T into B_T1 (4x4) and B_T2 (4x2)
    # B_T1 is B[0:4, 0:4]
    # B_T2 is B[0:4, 4:6]
    
    # --- Part 2a: A_L * B_T1 (4x4 * 4x4) ---
    # Use Strassen for 4x4x4
    # Blocks of A_L:
    # A00: A[0:2, 0:2]
    # A01: A[0:2, 2:4]
    # A10: A[2:4, 0:2]
    # A11: A[2:4, 2:4]
    
    # Blocks of B_T1:
    # B00: B[0:2, 0:2]
    # B01: B[0:2, 2:4]
    # B10: B[2:4, 0:2]
    # B11: B[2:4, 2:4]
    
    # Strassen M1..M7 are 2x2x2 multiplications
    # We need to map the linear combinations to the global indices.
    # A linear combination of A elements is a vector in U-space.
    # A linear combination of B elements is a vector in V-space.
    # The product is a bilinear form.
    # Since we are using the rank-1 decomposition of the sub-algorithm,
    # we can just generate the rank-1 terms for each 2x2x2 sub-problem.
    
    def strassen_2x2x2(a_indices, b_indices, c_indices):
        """
        a_indices: list of 4 u-indices for A block [[a00, a01], [a10, a11]]
        b_indices: list of 4 v-indices for B block [[b00, b01], [b10, b11]]
        c_indices: list of 4 w-indices for C block [[c00, c01], [c10, c11]]
        """
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # M1 = (a00 + a11)(b00 + b11)
        # This is a bilinear form. In the rank-1 decomposition of 2x2x2,
        # we don't just multiply sums. We use the 7 multiplications.
        # The 7 multiplications are:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01+b11)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Each mk is a product of two linear forms.
        # To decompose into rank-1 terms, we expand the linear forms?
        # No, the "multiplication" m1 is itself a bilinear operation on the coefficients.
        # However, in the context of the full tensor, m1 corresponds to a sub-tensor.
        # The rank of 2x2x2 is 7. This means we can represent the 2x2x2 tensor as sum of 7 rank-1 tensors.
        # The standard Strassen decomposition IS that rank-7 decomposition.
        # So we just need to add the 7 terms corresponding to m1..m7.
        # BUT, m1 is a product of two linear combinations.
        # Is m1 a rank-1 tensor in the global space?
        # Yes, if we view the coefficients of the linear combination as part of the U and V vectors.
        # For m1:
        # U_vector has 1 at a00 and 1 at a11.
        # V_vector has 1 at b00 and 1 at b11.
        # W_vector?
        # The contribution of m1 to C is:
        # C00 += m1
        # C01 += 0
        # C10 += 0
        # C11 += m1
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C00 and +1 to C11.
        # M2 contributes +1 to C10 and -1 to C11.
        # M3 contributes +1 to C01 and +1 to C11.
        # M4 contributes +1 to C00 and +1 to C10.
        # M5 contributes -1 to C00 and +1 to C01.
        # M6 contributes +1 to C11.
        # M7 contributes +1 to C00.
        
        # So for each Mk, we define U_vec, V_vec, and W_vec (which has coefficients in the C block).
        # We add a term for each non-zero entry in W_vec.
        
        # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (coeffs for c00, c01, c10, c11)
        # Actually, W_vec is a vector in the C-space.
        # We add (U_vec, V_vec, w_idx, coeff) for each w_idx where W_vec is non-zero.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: list of 4 coeffs for a00, a01, a10, a11
            # v_coeffs: list of 4 coeffs for b00, b01, b10, b11
            # w_coeffs: list of 4 coeffs for c00, c01, c10, c11
            for i in range(4):
                if u_coeffs[i] != 0:
                    for j in range(4):
                        if v_coeffs[j] != 0:
                            # This is not a single rank-1 term in the global sense if we expand.
                            # But the Strassen algorithm defines the rank-1 terms as the products of the linear forms.
                            # So the "term" is defined by the linear form in A and the linear form in B.
                            # The output is a linear combination of C.
                            # So we add a term:
                            # U has u_coeffs at the corresponding indices.
                            # V has v_coeffs at the corresponding indices.
                            # W has w_coeffs at the corresponding indices.
                            # This is a valid rank-1 term in the tensor space?
                            # Yes. A rank-1 tensor is u \otimes v \otimes w.
                            # Here u is a vector in U-space, v in V-space, w in W-space.
                            # The term is sum_{a,b,c} u_a v_b w_c e_a e_b e_c.
                            # This is exactly what we want.
                            pass
        
        # We need to be careful. The "term" in the decomposition is a single entry in the list of rank-1 tensors.
        # Each such tensor is defined by vectors u, v, w.
        # So for M1, u = [1, 0, 0, 1] (at indices a00, a01, a10, a11).
        # v = [1, 0, 0, 1] (at indices b00, b01, b10, b11).
        # w = [1, 0, 0, 1] (at indices c00, c01, c10, c11).
        # We add this triple to our list of terms.
        
        terms.append((a_indices, u_coeffs, b_indices, v_coeffs, c_indices, w_coeffs))
        
        # M1
        add_strassen_term([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
        # M2
        add_strassen_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])
        # M3
        add_strassen_term([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1])
        # M4
        add_strassen_term([0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0])
        # M5
        add_strassen_term([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
        # M6
        add_strassen_term([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1])
        # M7
        add_strassen_term([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])

    # We need to call strassen_2x2x2 for the 4x4x4 problem.
    # But the 4x4x4 problem uses 7 calls to 2x2x2.
    # The inputs to these 2x2x2 calls are linear combinations of the 4x4 blocks.
    # This gets complicated to track indices.
    
    # Alternative: Just hardcode the 49 terms for 4x4x4 and 28 terms for 4x4x2.
    # Or write a generic solver.
    
    # Given the complexity, I will implement the rank 105 version first (using schoolbook for 4x4x4 part? No, 49 is better).
    # I will implement the Strassen logic properly.
    
    # Let's refine the helper.
    
    return None # Placeholder
