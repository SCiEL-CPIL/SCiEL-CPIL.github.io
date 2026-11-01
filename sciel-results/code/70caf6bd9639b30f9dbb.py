"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U size: n*m = 8
    # V size: m*p = 24
    # W size: p*n = 12
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 6 sub-problems
    # Each sub-problem is defined by:
    # A_sub: 2x2 matrix from A. A is 2x4.
    #   A indices: (0, c0), (0, c1), (1, c0), (1, c1)
    # B_sub: 2x2 matrix from B. B is 4x6.
    #   B indices: (r0, d0), (r0, d1), (r1, d0), (r1, d1)
    # C_sub: 2x2 matrix from C. C is 2x6.
    #   C indices: (0, e0), (0, e1), (1, e0), (1, e1)
    
    subproblems = [
        # (A_c0, A_c1, B_r0, B_r1, B_d0, B_d1, C_e0, C_e1)
        # Sub-problem 1: A_L B_T0
        (0, 1, 0, 1, 0, 1, 0, 1),
        # Sub-problem 2: A_L B_T1
        (0, 1, 0, 1, 2, 3, 2, 3),
        # Sub-problem 3: A_L B_T2
        (0, 1, 0, 1, 4, 5, 4, 5),
        # Sub-problem 4: A_R B_B0
        (2, 3, 2, 3, 0, 1, 0, 1),
        # Sub-problem 5: A_R B_B1
        (2, 3, 2, 3, 2, 3, 2, 3),
        # Sub-problem 6: A_R B_B2
        (2, 3, 2, 3, 4, 5, 4, 5),
    ]
    
    # Strassen coefficients for the 7 multiplications
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    
    # U coefficients for a, b, c, d in m1..m7
    # a is (0,0), b is (0,1), c is (1,0), d is (1,1)
    # U indices: 
    # a: 0*4 + A_c0
    # b: 0*4 + A_c1
    # c: 1*4 + A_c0
    # d: 1*4 + A_c1
    
    # V coefficients for e, f, g, h in m1..m7
    # e is (0,0), f is (0,1), g is (1,0), h is (1,1) relative to sub-block
    # Global B indices:
    # e: B_r0*6 + B_d0
    # f: B_r0*6 + B_d1
    # g: B_r1*6 + B_d0
    # h: B_r1*6 + B_d1
    
    # W coefficients for C00, C01, C10, C11
    # C indices:
    # C00: C_e0*2 + 0
    # C01: C_e1*2 + 0  -- Wait. C is 2x6.
    # C[r, c]. r in 0..1, c in 0..5.
    # W index is k*n + i. Here k is row of C (0..5? No, C is 2x6).
    # Let's re-read the trace convention.
    # "W[k*n+i,r]" where n=2, p=6.
    # Usually C is n x p. So rows 0..n-1, cols 0..p-1.
    # The prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # In the seed: `for k in range(p): for i in range(n): W[k*n+i][column] = 1`.
    # This implies W index corresponds to (k, i) where k is the column of C? No.
    # Seed:
    # `U[i*m+j]` -> i (row A), j (col A/row B)
    # `V[j*p+k]` -> j (row B/col A), k (col B)
    # `W[k*n+i]` -> k (col B/col C), i (row A/row C)
    # So W index is `col_C * n + row_C`.
    # Let's verify:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Term is U[i, j, r] V[j, k, r] W[k, i, r].
    # If W index is `k*n + i`, then `W[k*n+i]` corresponds to C[i, k].
    # Yes.
    
    # So for C_sub 2x2 block at columns e0, e1:
    # C00 is C[0, e0] -> W index e0*2 + 0
    # C01 is C[0, e1] -> W index e1*2 + 0
    # C10 is C[1, e0] -> W index e0*2 + 1
    # C11 is C[1, e1] -> W index e1*2 + 1
    
    for (A_c0, A_c1, B_r0, B_r1, B_d0, B_d1, C_e0, C_e1) in subproblems:
        
        # Global indices for A elements
        u_a = 0 * 4 + A_c0
        u_b = 0 * 4 + A_c1
        u_c = 1 * 4 + A_c0
        u_d = 1 * 4 + A_c1
        
        # Global indices for B elements
        v_e = B_r0 * 6 + B_d0
        v_f = B_r0 * 6 + B_d1
        v_g = B_r1 * 6 + B_d0
        v_h = B_r1 * 6 + B_d1
        
        # Global indices for C elements
        w_00 = C_e0 * 2 + 0
        w_01 = C_e1 * 2 + 0
        w_10 = C_e0 * 2 + 1
        w_11 = C_e1 * 2 + 1
        
        # Define the 7 Strassen terms
        # Each term is a tuple of (U_indices_with_coeffs, V_indices_with_coeffs, W_indices_with_coeffs)
        
        # m1 = (a+d)(e+h)
        U_1 = [(u_a, 1), (u_d, 1)]
        V_1 = [(v_e, 1), (v_h, 1)]
        W_1 = [(w_00, 1), (w_11, 1)]
        
        # m2 = (c+d)e
        U_2 = [(u_c, 1), (u_d, 1)]
        V_2 = [(v_e, 1)]
        W_2 = [(w_10, 1), (w_11, -1)]
        
        # m3 = a(f-h)
        U_3 = [(u_a, 1)]
        V_3 = [(v_f, 1), (v_h, -1)]
        W_3 = [(w_01, 1), (w_11, 1)]
        
        # m4 = d(g-e)
        U_4 = [(u_d, 1)]
        V_4 = [(v_g, 1), (v_e, -1)]
        W_4 = [(w_00, 1), (w_10, 1)]
        
        # m5 = (a+b)h
        U_5 = [(u_a, 1), (u_b, 1)]
        V_5 = [(v_h, 1)]
        W_5 = [(w_01, 1), (w_00, -1)]
        
        # m6 = (c-a)(e+g)
        U_6 = [(u_c, 1), (u_a, -1)]
        V_6 = [(v_e, 1), (v_g, 1)]
        W_6 = [(w_11, 1)]
        
        # m7 = (b-d)(f+h)
        U_7 = [(u_b, 1), (u_d, -1)]
        V_7 = [(v_f, 1), (v_h, 1)]
        W_7 = [(w_00, 1)]
        
        # Add to global matrices
        for U_coeff, V_coeff, W_coeff in [
            (U_1, V_1, W_1),
            (U_2, V_2, W_2),
            (U_3, V_3, W_3),
            (U_4, V_4, W_4),
            (U_5, V_5, W_5),
            (U_6, V_6, W_6),
            (U_7, V_7, W_7)
        ]:
            for idx, coeff in U_coeff:
                U[idx][col] += coeff
            for idx, coeff in V_coeff:
                V[idx][col] += coeff
            for idx, coeff in W_coeff:
                W[idx][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
