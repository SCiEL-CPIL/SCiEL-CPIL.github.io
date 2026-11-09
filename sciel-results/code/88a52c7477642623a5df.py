"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # 23 terms for 3x3x3 decomposition (Strassen-based)
    # For a single 3x3x3 block, u[i*m+j], v[j*p+k], w[k*n+i]
    # where i,j,k in 0,1,2 and m=p=n=3 for the sub-problem
    
    # Each term is (u_dict, v_dict, w_dict) for the 3x3x3 sub-problem
    # u indices: i*3+j, v indices: j*3+k, w indices: k*3+i
    
    terms_3x3x3 = [
        # 7 Strassen terms for 2x2x2 (i,j,k in {0,1})
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(1,0): 1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),
        # 16 terms involving index 2
        # Schoolbook for terms with at least one index = 2
        # A[i,2]*B[2,k] for i in 0,1,2 and k in 0,1,2 (9 terms)
        ({(0,2): 1}, {(2,0): 1}, {(0,0): 1}),
        ({(0,2): 1}, {(2,1): 1}, {(1,0): 1}),
        ({(0,2): 1}, {(2,2): 1}, {(2,0): 1}),
        ({(1,2): 1}, {(2,0): 1}, {(0,1): 1}),
        ({(1,2): 1}, {(2,1): 1}, {(1,1): 1}),
        ({(1,2): 1}, {(2,2): 1}, {(2,1): 1}),
        ({(2,2): 1}, {(2,0): 1}, {(0,2): 1}),
        ({(2,2): 1}, {(2,1): 1}, {(1,2): 1}),
        ({(2,2): 1}, {(2,2): 1}, {(2,2): 1}),
        # A[i,0]*B[0,2], A[i,1]*B[1,2] for i in 0,1,2 (6 terms)
        ({(0,0): 1}, {(0,2): 1}, {(2,0): 1}),
        ({(1,0): 1}, {(0,2): 1}, {(2,1): 1}),
        ({(2,0): 1}, {(0,2): 1}, {(2,2): 1}),
        ({(0,1): 1}, {(1,2): 1}, {(2,0): 1}),
        ({(1,1): 1}, {(1,2): 1}, {(2,1): 1}),
        ({(2,1): 1}, {(1,2): 1}, {(2,2): 1}),
    ]
    
    assert len(terms_3x3x3) == 23, f"Expected 23 terms, got {len(terms_3x3x3)}"
    
    # Apply the 23-term decomposition to two blocks:
    # Block 0: columns 0,1,2 of B (k in 0,1,2)
    # Block 1: columns 3,4,5 of B (k in 3,4,5)
    
    for block in range(2):
        base_col = block * 23
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms_3x3x3):
            col = base_col + t_idx
            # u: i*m + j where i,j in 0,1,2 (m=3)
            for (i, j), val in u_dict.items():
                U[i * m + j][col] += val
            # v: j*p + k where j in 0,1,2, k in 0,1,2 for block 0, or 3,4,5 for block 1
            for (j, k_sub), val in v_dict.items():
                k = k_sub + block * 3
                V[j * p + k][col] += val
            # w: k*n + i where k in 0,1,2, i in 0,1,2 (n=3)
            for (k, i), val in w_dict.items():
                W[k * n + i][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
