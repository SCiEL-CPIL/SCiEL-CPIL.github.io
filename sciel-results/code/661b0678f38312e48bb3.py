"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-14 Strassen-based decomposition for 2x2 * 2x4
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for a single 2x2 block
    # U factors (coeffs of a00, a01, a10, a11)
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V factors for Left Block (b00, b01, b10, b11) mapped to indices 0,1,4,5
    # Indices: 0:b00, 1:b01, 2:b02, 3:b03, 4:b10, 5:b11, 6:b12, 7:b13
    V_strassen_L = [
        [1, 0, 0, 0, 0, 1, 0, 0], # M1
        [1, 0, 0, 0, 0, 0, 0, 0], # M2
        [0, 1, 0, 0, 0, -1, 0, 0],# M3
        [1, 0, 0, 0, -1, 0, 0, 0],# M4
        [0, 0, 0, 0, 0, 1, 0, 0], # M5
        [1, 1, 0, 0, 0, 0, 0, 0], # M6
        [0, 1, 0, 0, 1, 0, 0, 0]  # M7
    ]
    
    # V factors for Right Block (b02, b03, b12, b13) mapped to indices 2,3,6,7
    V_strassen_R = [
        [0, 0, 1, 0, 0, 0, 0, 1], # M1
        [0, 0, 1, 0, 0, 0, 0, 0], # M2
        [0, 0, 0, 1, 0, 0, 0, -1],# M3
        [0, 0, 1, 0, 0, 0, -1, 0],# M4
        [0, 0, 0, 0, 0, 0, 0, 1], # M5
        [0, 0, 1, 1, 0, 0, 0, 0], # M6
        [0, 0, 0, 1, 0, 0, 1, 0]  # M7
    ]
    
    # W factors for Left Block (c00, c10, c01, c11) mapped to indices 0,1,2,3
    # Indices: 0:c00, 1:c10, 2:c01, 3:c11, 4:c02, 5:c12, 6:c03, 7:c13
    W_strassen_L = [
        [1, 0, 0, 1, 0, 0, 0, 0], # M1
        [0, 1, 0, -1, 0, 0, 0, 0],# M2
        [0, 0, 1, 1, 0, 0, 0, 0], # M3
        [-1, 1, 0, 0, 0, 0, 0, 0],# M4
        [1, 0, 1, 0, 0, 0, 0, 0], # M5
        [0, 0, 0, 1, 0, 0, 0, 0], # M6
        [-1, 0, 0, 0, 0, 0, 0, 0] # M7
    ]
    
    # W factors for Right Block (c02, c12, c03, c13) mapped to indices 4,5,6,7
    W_strassen_R = [
        [0, 0, 0, 0, 1, 0, 0, 1], # M1
        [0, 0, 0, 0, 0, 1, 0, -1],# M2
        [0, 0, 0, 0, 0, 0, 1, 1], # M3
        [0, 0, 0, 0, -1, 1, 0, 0],# M4
        [0, 0, 0, 0, 1, 0, 1, 0], # M5
        [0, 0, 0, 0, 0, 0, 0, 1], # M6
        [0, 0, 0, 0, -1, 0, 0, 0] # M7
    ]
    
    # Fill U
    for r in range(7):
        for i in range(4):
            U[i][r] = U_strassen[r][i]
            U[i][r + 7] = U_strassen[r][i]
            
    # Fill V
    for r in range(7):
        for j in range(8):
            V[j][r] = V_strassen_L[r][j]
            V[j][r + 7] = V_strassen_R[r][j]
            
    # Fill W
    for r in range(7):
        for k in range(8):
            W[k][r] = W_strassen_L[r][k]
            W[k][r + 7] = W_strassen_R[r][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
