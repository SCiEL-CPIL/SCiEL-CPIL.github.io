DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    V_strassen = [
        [1, 0, 1, 0],  # M1
        [0, 1, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    coeff_c11 = [1, 0, 0, -1, 0, 1, -1]
    coeff_c12 = [0, 0, 1, 0, 1, -1, 1]
    coeff_c21 = [0, 1, 0, 1, -1, 0, 1]
    coeff_c22 = [1, 0, -1, 0, 0, -1, 0]
    
    for l in range(4):
        for s in range(7):
            col = l * 7 + s
            for i in range(4):
                U[i][col] = U_strassen[s][i]
            V[0 * 8 + l * 2 + 0][col] = V_strassen[s][0]
            V[0 * 8 + l * 2 + 1][col] = V_strassen[s][1]
            V[1 * 8 + l * 2 + 0][col] = V_strassen[s][2]
            V[1 * 8 + l * 2 + 1][col] = V_strassen[s][3]
            
            W[(l * 2 + 0) * 2 + 0][col] = coeff_c11[s]
            W[(l * 2 + 0) * 2 + 1][col] = coeff_c21[s]
            W[(l * 2 + 1) * 2 + 0][col] = coeff_c12[s]
            W[(l * 2 + 1) * 2 + 1][col] = coeff_c22[s]
            
    return {"U": U, "V": V, "W": W}
