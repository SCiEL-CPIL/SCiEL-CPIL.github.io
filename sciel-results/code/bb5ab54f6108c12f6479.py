"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x4x4 multiplication into block components.
    # A is 3x4, B is 4x4.
    # Split A into A_L (3x3) and A_R (3x1).
    # Split B into B_LL (3x3), B_LR (3x1), B_RL (1x3), B_RR (1x1).
    # C = A_L * B_LL + A_L * B_LR + A_R * B_RL + A_R * B_RR
    
    # Helper to generate schoolbook decomposition for n x m x p
    def schoolbook(n_sub, m_sub, p_sub, offset_u, offset_v, offset_w):
        rank_sub = n_sub * m_sub * p_sub
        U_sub = [[0] * rank_sub for _ in range(n_sub * m_sub)]
        V_sub = [[0] * rank_sub for _ in range(m_sub * p_sub)]
        W_sub = [[0] * rank_sub for _ in range(p_sub * n_sub)]
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_sub[i * m_sub + j][col] = 1
                    V_sub[j * p_sub + k][col] = 1
                    W_sub[k * n_sub + i][col] = 1
                    col += 1
        return U_sub, V_sub, W_sub, rank_sub

    # Total rank is sum of ranks of sub-blocks
    # Block 1: A_L (3x3) * B_LL (3x3) -> 3x3
    U1, V1, W1, r1 = schoolbook(3, 3, 3, 0, 0, 0)
    # Block 2: A_L (3x3) * B_LR (3x1) -> 3x1
    U2, V2, W2, r2 = schoolbook(3, 3, 1, 0, 3, 0) # Note: V index offset for B_LR
    # Block 3: A_R (3x1) * B_RL (1x3) -> 3x3
    U3, V3, W3, r3 = schoolbook(3, 1, 3, 9, 0, 0) # Note: U index offset for A_R
    # Block 4: A_R (3x1) * B_RR (1x1) -> 3x1
    U4, V4, W4, r4 = schoolbook(3, 1, 1, 9, 3, 0) # Note: U index offset for A_R, V index offset for B_RR

    total_rank = r1 + r2 + r3 + r4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    # Block 1: Rows 0..8 (3*3)
    for i in range(3):
        for j in range(3):
            for c in range(r1):
                U[i * 4 + j][c] = U1[i * 3 + j][c]
    # Block 2: Rows 0..8 (3*3)
    for i in range(3):
        for j in range(3):
            for c in range(r2):
                U[i * 4 + j][r1 + c] = U2[i * 3 + j][c]
    # Block 3: Rows 12..14 (3*1, j=3)
    for i in range(3):
        for c in range(r3):
            U[i * 4 + 3][r1 + r2 + c] = U3[i * 1 + 0][c]
    # Block 4: Rows 12..14 (3*1, j=3)
    for i in range(3):
        for c in range(r4):
            U[i * 4 + 3][r1 + r2 + r3 + c] = U4[i * 1 + 0][c]

    # Fill V
    # Block 1: Cols 0..8 (3*3)
    for j in range(3):
        for k in range(3):
            for c in range(r1):
                V[j * 4 + k][c] = V1[j * 3 + k][c]
    # Block 2: Cols 0..2 (3*1, k=0..2? No, B_LR is 3x1, so k=0..2 is wrong. B_LR is columns 0..2 of B? No.
    # B is 4x4. B_LL is rows 0..2, cols 0..2. B_LR is rows 0..2, col 3.
    # So for V, j is 0..2, k is 3.
    for j in range(3):
        for c in range(r2):
            V[j * 4 + 3][r1 + c] = V2[j * 1 + 0][c]
    # Block 3: B_RL is row 3, cols 0..2.
    for k in range(3):
        for c in range(r3):
            V[3 * 4 + k][r1 + r2 + c] = V3[0 * 3 + k][c]
    # Block 4: B_RR is row 3, col 3.
    for c in range(r4):
        V[3 * 4 + 3][r1 + r2 + r3 + c] = V4[0 * 1 + 0][c]

    # Fill W
    # W indices are k*n + i. n=3.
    # Block 1: C is 3x3. k=0..2, i=0..2.
    for k in range(3):
        for i in range(3):
            for c in range(r1):
                W[k * 3 + i][c] = W1[k * 3 + i][c]
    # Block 2: C is 3x1. k=0..2, i=0..2. (Output col 0)
    for k in range(3):
        for i in range(3):
            for c in range(r2):
                W[k * 3 + i][r1 + c] = W2[k * 1 + i][c]
    # Block 3: C is 3x3. k=0..2, i=0..2. (Output cols 0..2)
    for k in range(3):
        for i in range(3):
            for c in range(r3):
                W[k * 3 + i][r1 + r2 + c] = W3[k * 3 + i][c]
    # Block 4: C is 3x1. k=0..2, i=0..2. (Output col 3)
    for k in range(3):
        for i in range(3):
            for c in range(r4):
                W[k * 3 + i][r1 + r2 + r3 + c] = W4[k * 1 + i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
