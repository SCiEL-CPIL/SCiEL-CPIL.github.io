"""Matrix multiplication 4x4x7 using Strassen-based tensor decomposition."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen 2x2x2 rank-7 matrices
    U_S = [
        [1, 1, 0, 1, 1, -1, 0],   # a11
        [0, 0, 1, 0, 1, 0, -1],   # a12
        [0, 0, 0, 0, 0, 1, 0],    # a21
        [0, 1, -1, 0, 0, 0, 1],   # a22
    ]
    V_S = [
        [1, 0, 0, 0, 1, 1, -1],   # b11
        [0, 0, 0, 1, 0, 1, 0],    # b12
        [0, 0, 1, 0, 0, 0, 1],    # b21
        [1, 1, 1, -1, 0, 0, 0],   # b22
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],   # c11
        [0, 0, 1, 0, 1, 0, 0],    # c12
        [0, 1, 0, 1, 0, 0, 0],    # c21
        [1, -1, 1, 0, 0, 1, 0],   # c22
    ]
    
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # PART 1: 4x4x4 part (B columns 0-3), rank 49
    # PART 2: 4x4x3 part (B columns 4-6), rank 48
    # Total rank: 97
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # PART 1: Strassen ⊗ Strassen for 4x4x4
    # A_flat[i*4+j], B_flat[j*4+k] for k in {0,1,2,3}, C_flat[k*4+i]
    for r in range(7):
        for s in range(7):
            col = r*7 + s
            for a_idx in range(4):
                for b_idx in range(4):
                    flat_A = a_idx*4 + b_idx
                    flat_B = a_idx*4 + b_idx  # Same indexing for B (4x4)
                    flat_C = a_idx*4 + b_idx  # Same for C (4x4)
                    U[flat_A][col] = U_S[a_idx][r] * U_S[b_idx][s]
                    V[flat_B][col] = V_S[a_idx][r] * V_S[b_idx][s]
                    W[flat_C][col] = W_S[a_idx][r] * W_S[b_idx][s]
    
    # PART 2: Schoolbook for 4x4x3 (B columns 4-6)
    # A_flat[i*4+l], B_flat[l*4+(k-4)], C_flat[k*4+i]
    col = rank1
    for i in range(4):
        for l in range(4):
            for k in range(3):
                k_actual = k + 4
                U[i*4+l][col] = 1
                V[l*4+k][col] = 1
                W[k_actual*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
