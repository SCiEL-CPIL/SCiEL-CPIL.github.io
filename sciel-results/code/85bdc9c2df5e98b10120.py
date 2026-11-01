import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    A_coeffs = []
    B_coeffs = []
    C_coeffs = []
    
    # M1
    A1 = np.zeros((4, 4), dtype=int); A1[0:2, 0:2] = 1; A1[2:4, 2:4] = 1
    B1 = np.zeros((4, 4), dtype=int); B1[0:2, 0:2] = 1; B1[2:4, 2:4] = 1
    C1 = np.zeros((4, 4), dtype=int); C1[0:2, 0:2] = 1
    A_coeffs.append(A1); B_coeffs.append(B1); C_coeffs.append(C1)
    
    # M2
    A2 = np.zeros((4, 4), dtype=int); A2[2:4, 0:2] = 1; A2[2:4, 2:4] = 1
    B2 = np.zeros((4, 4), dtype=int); B2[0:2, 0:2] = 1
    C2 = np.zeros((4, 4), dtype=int); C2[2:4, 0:2] = 1
    A_coeffs.append(A2); B_coeffs.append(B2); C_coeffs.append(C2)
    
    # M3
    A3 = np.zeros((4, 4), dtype=int); A3[0:2, 0:2] = 1
    B3 = np.zeros((4, 4), dtype=int); B3[0:2, 2:4] = 1; B3[2:4, 2:4] = -1
    C3 = np.zeros((4, 4), dtype=int); C3[0:2, 2:4] = 1
    A_coeffs.append(A3); B_coeffs.append(B3); C_coeffs.append(C3)
    
    # M4
    A4 = np.zeros((4, 4), dtype=int); A4[2:4, 2:4] = 1
    B4 = np.zeros((4, 4), dtype=int); B4[2:4, 0:2] = 1; B4[0:2, 0:2] = -1
    C4 = np.zeros((4, 4), dtype=int); C4[0:2, 0:2] = 1; C4[2:4, 0:2] = 1
    A_coeffs.append(A4); B_coeffs.append(B4); C_coeffs.append(C4)
    
    # M5
    A5 = np.zeros((4, 4), dtype=int); A5[0:2, 0:2] = 1; A5[0:2, 2:4] = 1
    B5 = np.zeros((4, 4), dtype=int); B5[2:4, 2:4] = 1
    C5 = np.zeros((4, 4), dtype=int); C5[0:2, 0:2] = -1; C5[0:2, 2:4] = 1
    A_coeffs.append(A5); B_coeffs.append(B5); C_coeffs.append(C5)
    
    # M6
    A6 = np.zeros((4, 4), dtype=int); A6[2:4, 0:2] = 1; A6[0:2, 0:2] = -1
    B6 = np.zeros((4, 4), dtype=int); B6[0:2, 0:2] = 1; B6[0:2, 2:4] = 1
    C6 = np.zeros((4, 4), dtype=int); C6[2:4, 2:4] = 1
    A_coeffs.append(A6); B_coeffs.append(B6); C_coeffs.append(C6)
    
    # M7
    A7 = np.zeros((4, 4), dtype=int); A7[0:2, 2:4] = 1; A7[2:4, 2:4] = -1
    B7 = np.zeros((4, 4), dtype=int); B7[2:4, 0:2] = 1; B7[2:4, 2:4] = 1
    C7 = np.zeros((4, 4), dtype=int); C7[0:2, 0:2] = 1
    A_coeffs.append(A7); B_coeffs.append(B7); C_coeffs.append(C7)
    
    block_indices = []
    for i in range(16):
        r = i // 4
        c = i % 4
        if r < 2 and c < 2: block_indices.append(0)
        elif r < 2 and c >= 2: block_indices.append(1)
        elif r >= 2 and c < 2: block_indices.append(2)
        else: block_indices.append(3)
        
    for k in range(7):
        for s in range(7):
            col = k * 7 + s
            for i in range(16):
                U[i, col] = U_2[block_indices[i], s] * A_coeffs[k][i]
                V[i, col] = V_2[block_indices[i], s] * B_coeffs[k][i]
                W[i, col] = W_2[block_indices[i], s] * C_coeffs[k][i]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
