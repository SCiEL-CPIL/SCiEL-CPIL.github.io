def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_scalar(u_idx, v_idx, w_idx, coeff):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        u_vec[u_idx] = coeff
        v_vec[v_idx] = coeff
        w_vec[w_idx] = coeff
        add_tensor(u_vec, v_vec, w_vec)

    # Strassen 2x2x2
    # A: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # B: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # C: (k0,i0), (k0,i1), (k1,i0), (k1,i1)
    def strassen_2x2x2(i0, j0, i1, j1, k0, k1):
        # Note: A is n x m, so A[i, j]. Index in U: i*m + j.
        # B is m x p, so B[j, k]. Index in V: j*p + k.
        # C is p x n, so C[k, i]. Index in W: k*n + i.
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # We need to represent the sum in U and V.
        # U has A[i0,j0] and A[i1,j1] with coeff 1.
        # V has B[j0,k0] and B[j1,k1] with coeff 1.
        # W has C[k0,i0] coeff 1, C[k1,i1] coeff 1, C[k0,i1] coeff -1, C[k1,i0] coeff 1.
        
        def add_combined(u_indices_coeffs, v_indices_coeffs, w_indices_coeffs):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            for idx, c in u_indices_coeffs:
                u_vec[idx] = c
            for idx, c in v_indices_coeffs:
                v_vec[idx] = c
            for idx, c in w_indices_coeffs:
                w_vec[idx] = c
            add_tensor(u_vec, v_vec, w_vec)

        # M1
        u_idx_a00 = i0 * m + j0
        u_idx_a11 = i1 * m + j1
        v_idx_b00 = j0 * p + k0
        v_idx_b11 = j1 * p + k1
        w_idx_c00 = k0 * n + i0
        w_idx_c11 = k1 * n + i1
        w_idx_c01 = k0 * n + i1
        w_idx_c10 = k1 * n + i0
        
        add_combined(
            [(u_idx_a00, 1), (u_idx_a11, 1)],
            [(v_idx_b00, 1), (v_idx_b11, 1)],
            [(w_idx_c00, 1), (w_idx_c11, 1), (w_idx_c01, -1), (w_idx_c10, 1)]
        )

        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        u_idx_a10 = i1 * m + j0
        add_combined(
            [(u_idx_a10, 1), (u_idx_a11, 1)],
            [(v_idx_b00, 1)],
            [(w_idx_c10, 1)]
        )

        # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        u_idx_b01 = j0 * p + k1
        u_idx_b11 = j1 * p + k1 # Wait, v_idx_b11 is j1*p+k1
        add_combined(
            [(u_idx_a00, 1)],
            [(u_idx_b01, 1), (v_idx_b11, -1)],
            [(w_idx_c01, 1)]
        )

        # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        v_idx_b10 = j1 * p + k0
        add_combined(
            [(u_idx_a11, 1)],
            [(v_idx_b10, 1), (v_idx_b00, -1)],
            [(w_idx_c10, 1)]
        )

        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        u_idx_a01 = i0 * m + j1
        add_combined(
            [(u_idx_a00, 1), (u_idx_a01, 1)],
            [(v_idx_b11, 1)],
            [(w_idx_c11, 1)]
        )

        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        add_combined(
            [(u_idx_a10, 1), (u_idx_a00, -1)],
            [(v_idx_b00, 1), (u_idx_b01, 1)],
            [(w_idx_c11, 1)]
        )

        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        add_combined(
            [(u_idx_a01, 1), (u_idx_a11, -1)],
            [(v_idx_b10, 1), (v_idx_b11, 1)],
            [(w_idx_c11, 1)]
        )

    # Block 1: A[0..1, 0..1] * B[0..1, 0..5]
    # A indices: i in 0,1; j in 0,1
    # B indices: j in 0,1; k in 0..5
    # C indices: k in 0..5; i in 0,1
    
    # Split B columns into pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        k0 = k_start
        k1 = k_start + 1
        # A is 2x2 (i=0,1; j=0,1)
        # B is 2x2 (j=0,1; k=k0,k1)
        strassen_2x2x2(0, 0, 1, 1, k0, k1)

    # Block 2: A[0..1, 2..4] * B[2..4, 0..5]
    # A indices: i in 0,1; j in 2,3,4
    # B indices: j in 2,3,4; k in 0..5
    # C indices: k in 0..5; i in 0,1
    
    # Split B columns into pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        k0 = k_start
        k1 = k_start + 1
        # Subproblem: 2x3 x 2
        # A_sub: 2x3 (i=0,1; j=2,3,4)
        # B_sub: 3x2 (j=2,3,4; k=k0,k1)
        
        # Decompose into 2x2x2 (Strassen) + 2x1x2 (Schoolbook)
        # Part 1: A[0..1, 2..3] * B[2..3, k0..k1]
        strassen_2x2x2(0, 2, 1, 3, k0, k1)
        
        # Part 2: A[0..1, 4] * B[4, k0..k1]
        # This is 2x1 x 2
        # C[0, k0] += A[0,4] * B[4, k0]
        # C[0, k1] += A[0,4] * B[4, k1]
        # C[1, k0] += A[1,4] * B[4, k0]
        # C[1, k1] += A[1,4] * B[4, k1]
        
        # 4 terms
        # Term 1: A[0,4] * B[4, k0] -> C[0, k0]
        add_scalar(0 * m + 4, 4 * p + k0, k0 * n + 0, 1)
        # Term 2: A[0,4] * B[4, k1] -> C[0, k1]
        add_scalar(0 * m + 4, 4 * p + k1, k1 * n + 0, 1)
        # Term 3: A[1,4] * B[4, k0] -> C[1, k0]
        add_scalar(1 * m + 4, 4 * p + k0, k0 * n + 1, 1)
        # Term 4: A[1,4] * B[4, k1] -> C[1, k1]
        add_scalar(1 * m + 4, 4 * p + k1, k1 * n + 1, 1)

    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    # Wait, U_cols is list of vectors. We need U to be list of rows?
    # Seed program: U is list of lists, U[i][j] is element.
    # My U_cols: list of vectors (length n*m).
    # So U[i][col] = U_cols[col][i].
    # So U = [[U_cols[c][i] for c in range(len(U_cols))] for i in range(n*m)]
    
    if not U_cols:
        return {"U": [], "V": [], "W": []}
        
    rank = len(U_cols)
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n * m)]
    V = [[V_cols[c][i] for c in range(rank)] for i in range(m * p)]
    W = [[W_cols[c][i] for c in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
