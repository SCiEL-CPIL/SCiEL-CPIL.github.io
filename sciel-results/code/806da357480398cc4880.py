"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Rank-28 decomposition using 4 independent Strassen (rank-7) decompositions
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2x2: (U_A, V_B, W_C)
    # A=(a0,a1,a2,a3), B=(b0,b1,b2,b3), C=(c0,c1,c2,c3)
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    # Sub-muls: (A_indices, B_indices, C_indices)
    # A: 2x4, index = i*4+j
    # B: 4x4, index = j*4+k
    # C: 2x4, index = k*2+i (W[k*n+i,r] convention)
    sub_muls = [
        # A1*B11 -> C1 (cols 0,1)
        ([0,1,4,5], [0,1,4,5], [0,1,4,5]),
        # A2*B21 -> C1 (cols 0,1)
        ([2,3,6,7], [8,9,12,13], [0,1,4,5]),
        # A1*B12 -> C2 (cols 2,3)
        ([0,1,4,5], [2,3,6,7], [2,3,6,7]),
        # A2*B22 -> C2 (cols 2,3)
        ([2,3,6,7], [10,11,14,15], [2,3,6,7]),
    ]
    
    col = 0
    for (A_idx, B_idx, C_idx) in sub_muls:
        for (U_A, V_B, W_C) in strassen_terms:
            for i, a_pos in enumerate(A_idx):
                U[a_pos][col] = U_A[i]
            for j, b_pos in enumerate(B_idx):
                V[b_pos][col] = V_B[j]
            for k, c_pos in enumerate(C_idx):
                W[c_pos][col] = W_C[k]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
