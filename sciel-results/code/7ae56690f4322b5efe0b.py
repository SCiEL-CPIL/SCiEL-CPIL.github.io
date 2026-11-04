def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal col
        U[u_idx][col] = coeff_u
        V[v_idx][col] = coeff_v
        W[w_idx][col] = coeff_w
        col += 1

    # Part 1: C_2 = A_2 B (1x4 * 4x5) -> Rank 20
    # A_2 is row 2 of A (index 2)
    # B is 4x5
    for j in range(4):
        for k in range(5):
            u_idx = 2 * 4 + j  # Row 2, col j of A
            v_idx = j * 5 + k  # Row j, col k of B
            w_idx = k * 3 + 2  # Row k, col 2 of C
            add_term(u_idx, v_idx, w_idx)
            
    # Part 2: C_1 = A_1 B (2x4 * 4x5) -> Rank 36
    # A_1 is rows 0,1 of A.
    # Split A_1 into A_11 (2x2, cols 0,1) and A_12 (2x2, cols 2,3)
    # Split B into B1 (2x5, rows 0,1) and B2 (2x5, rows 2,3)
    
    # C_11 = A_11 B1 (2x2 * 2x5) -> Rank 18
    # A_11 indices: u (0,0), (0,1), (1,0), (1,1)
    # B1 indices: v (0,0..4), (1,0..4)
    # C_11 indices: w (0,0..4), (1,0..4)
    
    # Split B1 into B11 (2x2, cols 0,1) and B12 (2x1, col 2? No, 2x5 split into 2x4 + 2x1)
    # B11: cols 0,1,2,3? No, 2x4.
    # Let's split B1 (2x5) into B1a (2x4) and B1b (2x1).
    # B1a cols 0,1,2,3. B1b col 4.
    
    # C_11a = A_11 B1a (2x2 * 2x4) -> Rank 14
    # Split B1a (2x4) into B1a1 (2x2, cols 0,1) and B1a2 (2x2, cols 2,3)
    # C_11a1 = A_11 B1a1 (2x2 * 2x2) -> Rank 7 (Strassen)
    # C_11a2 = A_11 B1a2 (2x2 * 2x2) -> Rank 7 (Strassen)
    
    # C_11b = A_11 B1b (2x2 * 2x1) -> Rank 4
    # C_11 = C_11a1 + C_11a2 + C_11b
    
    # We need a helper for 2x2x2 Strassen
    def add_strassen(u_base, v_base, w_base, v_col_offset=0):
        # u_base: (i, j) offset in U
        # v_base: (j, k) offset in V
        # w_base: (k, i) offset in W
        # v_col_offset: offset for k in V
        nonlocal col
        
        # Indices
        # A: u(i, j)
        # B: v(j, k)
        # C: w(k, i)
        
        # m1 = (a+d)(e+h)
        # a=u(0,0), d=u(1,1), e=v(0, v_col_offset), h=v(1, v_col_offset+1)
        # This is not a single term, it's a sum.
        # We need to expand the Strassen formula into elementary terms?
        # No, Strassen terms are bilinear.
        # m1 = (A_00 + A_11) * (B_00 + B_11)
        # This is 4 terms: A_00 B_00, A_00 B_11, A_11 B_00, A_11 B_11.
        # But Strassen reduces 8 to 7.
        # The 7 multiplications are:
        # m1 = (A_00 + A_11) * (B_00 + B_11)
        # m2 = (A_10 + A_11) * B_00
        # m3 = A_00 * (B_01 - B_11)
        # m4 = A_11 * (B_10 - B_00)
        # m5 = (A_00 + A_01) * B_11
        # m6 = (A_10 - A_00) * (B_00 + B_01)
        # m7 = (A_01 - A_11) * (B_10 + B_11)
        
        # Each m is a product of a linear combination of U and V.
        # In the tensor decomposition, each m corresponds to ONE column in U, V, W?
        # No, the decomposition is $\sum_{c=1}^R u_c \otimes v_c \otimes w_c$.
        # If $u_c$ is a vector, $v_c$ is a vector, $w_c$ is a vector.
        # The term is $\sum_{i,j,k} u_{c,i,j} v_{c,j,k} w_{c,k,i}$.
        # For Strassen, $m_1$ is one "multiplication".
        # $u_{c}$ has non-zeros at (0,0) and (1,1) with value 1.
        # $v_{c}$ has non-zeros at (0,0) and (1,1) with value 1.
        # $w_{c}$ has non-zeros at (0,0), (0,1), (1,0), (1,1) with values 1, -1, 1, -1?
        # Let's check.
        # $m_1 = (A_{00} + A_{11})(B_{00} + B_{11})$.
        # Contribution to $C_{00}$: $+1 \cdot m_1$.
        # Contribution to $C_{01}$: $-1 \cdot m_1$? No.
        # $C_{00} = m_1 + m_4 - m_5 + m_7$.
        # $C_{01} = m_3 + m_5$.
        # $C_{10} = m_2 + m_4$.
        # $C_{11} = m_1 - m_2 + m_3 + m_6$.
        
        # So for m1:
        # u: (0,0)=1, (1,1)=1.
        # v: (0,0)=1, (1,1)=1.
        # w: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1.
        
        # We will implement a function to add a Strassen block.
        
        def add_strassen_term(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i, coeffs_u, coeffs_v, coeffs_w):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    if coeffs_u[di][dj] != 0:
                        U[u_off_i + di][u_off_j + dj][col] = coeffs_u[di][dj]
            for dj in range(2):
                for dk in range(2):
                    if coeffs_v[dj][dk] != 0:
                        V[v_off_j + dj][v_off_k + dk][col] = coeffs_v[dj][dk]
            for dk in range(2):
                for di in range(2):
                    if coeffs_w[dk][di] != 0:
                        W[w_off_k + dk][w_off_i + di][col] = coeffs_w[dk][di]
            col += 1

        # m1
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]
        )
        # m2
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]
        )
        # m3
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]
        )
        # m4
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[0,0],[0,1]], [[1,-1],[0,0]], [[1,0],[1,0]]
        )
        # m5
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]
        )
        # m6
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]
        )
        # m7
        add_strassen_term(
            u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i,
            [[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,-1]]
        )

    # Call Strassen for E1: A_11 (2x2) * B1a1 (2x2)
    # A_11: u rows 0,1; u cols 0,1.
    # B1a1: v rows 0,1; v cols 0,1.
    # C: w rows 0,1; w cols 0,1 (relative to C_11 which is C rows 0,1, cols 0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Call Strassen for E2: A_11 (2x2) * B1a2 (2x2)
    # B1a2: v rows 0,1; v cols 2,3.
    # C: w rows 2,3; w cols 0,1.
    add_strassen(0, 0, 0, 2, 2, 0)
    
    # Call Rank 4 for C_11b: A_11 (2x2) * B1b (2x1)
    # B1b: v rows 0,1; v col 4.
    # C: w row 4; w cols 0,1.
    for i in range(2):
        for j in range(2):
            add_term(i * 4 + j, j * 5 + 4, 4 * 3 + i)
            
    # Now C_12 = A_12 B2 (2x2 * 2x5) -> Rank 18
    # A_12: u rows 0,1; u cols 2,3.
    # B2: v rows 2,3; v cols 0,4.
    
    # C_12a = A_12 B2a (2x2 * 2x4) -> Rank 14
    # B2a: v rows 2,3; v cols 0,3.
    # C_12a1 = A_12 B2a1 (2x2 * 2x2) -> Rank 7
    # B2a1: v rows 2,3; v cols 0,1.
    # C: w rows 0,1; w cols 2,3.
    add_strassen(0, 2, 2, 0, 0, 2)
    
    # C_12a2 = A_12 B2a2 (2x2 * 2x2) -> Rank 7
    # B2a2: v rows 2,3; v cols 2,3.
    # C: w rows 2,3; w cols 2,3.
    add_strassen(0, 2, 2, 2, 2, 2)
    
    # C_12b = A_12 B2b (2x2 * 2x1) -> Rank 4
    # B2b: v rows 2,3; v col 4.
    # C: w row 4; w cols 2,3.
    for i in range(2):
        for j in range(2):
            add_term(i * 4 + (j+2), (j+2) * 5 + 4, 4 * 3 + (i+2))

    return {"U": U, "V": V, "W": W}
