"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # 1 term for j=1
    # 3 blocks of 2x2x2, each using Strassen (rank 7)
    total_rank = 1 + 3 * 7
    rank = total_rank
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 component
    # u_coeffs: dict {(i, j): coeff}
    # v_coeffs: dict {(j, k): coeff}
    # w_coeffs: dict {(k, i): coeff}
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i][col] += val

    # Term 1: j=1
    # C_ik += A_i1 B_1k
    u1 = {(i, 1): 1.0 for i in range(n)}
    v1 = {(1, k): 1.0 for k in range(p)}
    w1 = {(k, i): 1.0 for k in range(p) for i in range(n)}
    add_term(u1, v1, w1, 0)
    
    # Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M5

    # We have 3 blocks of B, corresponding to k in {1,2}, {3,4}, {5,6}
    # In 0-indexed: k in {0,1}, {2,3}, {4,5}
    # A' corresponds to A[:, 1:3] (indices 1, 2 in 0-indexed, which are j=2,3 in 1-indexed)
    # So local A11 is A[0,1], A12 is A[0,2], A21 is A[1,1], A22 is A[1,2]
    
    col = 1
    for s in range(3):
        # k indices for this block
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # Local B indices
        # B11 = B[1, k0] (j=2, k=k0)
        # B12 = B[1, k1]
        # B21 = B[2, k0] (j=3, k=k0)
        # B22 = B[2, k1]
        
        # Define the 7 Strassen terms
        # Each term has u_coeffs, v_coeffs
        # And we need to map to W coeffs for C11, C12, C21, C22 of this block
        # C11 corresponds to C[0, k0] (i=0, k=k0)
        # C12 corresponds to C[0, k1] (i=0, k=k1)
        # C21 corresponds to C[1, k0] (i=1, k=k0)
        # C22 corresponds to C[1, k1] (i=1, k=k1)
        
        # M1
        u_m1 = {(0, 1): 1.0, (1, 2): 1.0} # A11, A22
        v_m1 = {(1, k0): 1.0, (2, k1): 1.0} # B11, B22
        w_m1 = {(k0, 0): 1.0, (k1, 1): 1.0} # C11, C22
        add_term(u_m1, v_m1, w_m1, col)
        col += 1
        
        # M2
        u_m2 = {(1, 1): 1.0, (1, 2): 1.0} # A21, A22
        v_m2 = {(1, k0): 1.0} # B11
        w_m2 = {(k0, 1): 1.0, (k1, 1): 1.0} # C21, C22
        add_term(u_m2, v_m2, w_m2, col)
        col += 1
        
        # M3
        u_m3 = {(0, 1): 1.0} # A11
        v_m3 = {(1, k1): 1.0, (2, k1): -1.0} # B12, -B22
        w_m3 = {(k1, 0): 1.0, (k1, 1): -1.0} # C12, -C22
        add_term(u_m3, v_m3, w_m3, col)
        col += 1
        
        # M4
        u_m4 = {(1, 2): 1.0} # A22
        v_m4 = {(2, k0): 1.0, (1, k0): -1.0} # B21, -B11
        w_m4 = {(k0, 0): -1.0, (k0, 1): 1.0} # -C11, C21
        add_term(u_m4, v_m4, w_m4, col)
        col += 1
        
        # M5
        u_m5 = {(0, 1): 1.0, (0, 2): 1.0} # A11, A12
        v_m5 = {(2, k1): 1.0} # B22
        w_m5 = {(k1, 0): 1.0, (k1, 1): 1.0} # C12, C22
        add_term(u_m5, v_m5, w_m5, col)
        col += 1
        
        # M6
        u_m6 = {(1, 1): 1.0, (0, 1): -1.0} # A21, -A11
        v_m6 = {(1, k0): 1.0, (1, k1): 1.0} # B11, B12
        w_m6 = {(k0, 0): 1.0} # C11
        add_term(u_m6, v_m6, w_m6, col)
        col += 1
        
        # M7
        u_m7 = {(0, 2): 1.0, (1, 2): -1.0} # A12, -A22
        v_m7 = {(2, k0): 1.0, (2, k1): 1.0} # B21, B22
        w_m7 = {(k0, 0): 1.0} # C11
        add_term(u_m7, v_m7, w_m7, col)
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
