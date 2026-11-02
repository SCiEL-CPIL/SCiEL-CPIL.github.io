"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_str[i*m+j, r], V_str[j*p+k, r], W_str[k*n+i, r] for 2x2x2
    # Here p=2 for the sub-blocks
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    U_str[0][0] = 1; U_str[3][0] = 1
    V_str[0][0] = 1; V_str[3][0] = 1
    W_str[0][0] = 1; W_str[3][0] = 1
    
    # r=1: M2
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[2][1] = 1; W_str[3][1] = -1
    
    # r=2: M3
    U_str[0][2] = 1
    V_str[1][2] = 1; V_str[3][2] = -1
    W_str[1][2] = 1; W_str[3][2] = 1
    
    # r=3: M4
    U_str[3][3] = 1
    V_str[2][3] = 1; V_str[0][3] = -1
    W_str[0][3] = 1; W_str[2][3] = 1
    
    # r=4: M5
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    W_str[1][4] = -1; W_str[3][4] = 1
    
    # r=5: M6
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    W_str[3][5] = 1
    
    # r=6: M7
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    W_str[0][6] = 1
    
    # Map to 2x2x4
    # Block 1: k in {0, 1}
    # Block 2: k in {2, 3}
    
    for i in range(2):
        for j in range(2):
            idx_U = i * 2 + j
            for r in range(7):
                U[idx_U][r] = U_str[idx_U][r]
                U[idx_U][r + 7] = U_str[idx_U][r]
                
    for j in range(2):
        for k in range(2):
            idx_V_block1 = j * 4 + k
            idx_V_block2 = j * 4 + (2 + k)
            idx_V_str = j * 2 + k
            for r in range(7):
                V[idx_V_block1][r] = V_str[idx_V_str][r]
                V[idx_V_block2][r + 7] = V_str[idx_V_str][r]
                
    for k in range(2):
        for i in range(2):
            idx_W_block1 = k * 2 + i
            idx_W_block2 = (2 + k) * 2 + i
            idx_W_str = k * 2 + i
            for r in range(7):
                W[idx_W_block1][r] = W_str[idx_W_str][r]
                W[idx_W_block2][r + 7] = W_str[idx_W_str][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
