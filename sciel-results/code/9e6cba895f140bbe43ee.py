import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A flattened: [a00, a01, a10, a11]
    # B flattened: [b00, b01, b10, b11]
    # C flattened: [c00, c01, c10, c11]
    
    U_strassen = [
        [-1, 0, 0, 1], # M1
        [ 1, 0, 0, 1], # M2
        [ 1, 1, 0, 0], # M3
        [-1, 0, 1, 0], # M4
        [ 0, 1, 0, 1], # M5
        [ 1, 0, -1, 0],# M6
        [ 0, 1, 0, -1] # M7
    ]
    
    V_strassen = [
        [ 1, 0, 0, -1],# M1
        [ 0, 0, 0, 1], # M2
        [ 1, 0, 0, 0], # M3
        [ 0, 1, 0, 1], # M4
        [ 0, 0, 1, 1], # M5
        [ 1, 1, 0, 0], # M6
        [ 1, 0, 1, 0]  # M7
    ]
    
    W_strassen = [
        [ 1, 0, 0, 1], # M1
        [ 0, 0, 1, -1],# M2
        [ 0, 1, 0, 1], # M3
        [ 1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [ 0, 0, 0, 1], # M6
        [ 1, 0, 0, 0]  # M7
    ]
    
    # Define blocks: (A_offsets, B_offsets, C_offsets)
    # A_offsets: list of 4 global indices for a00, a01, a10, a11
    # B_offsets: list of 4 global indices for b00, b01, b10, b11
    # C_offsets: list of 4 global indices for c00, c01, c10, c11
    
    blocks = [
        # Block 1: A1 B11 -> C11
        ([0, 1, 4, 5], [0, 1, 6, 7], [0, 2, 1, 3]),
        # Block 2: A2 B21 -> C11
        ([2, 3, 6, 7], [12, 13, 18, 19], [0, 2, 1, 3]),
        # Block 3: A1 B12 -> C21
        ([0, 1, 4, 5], [2, 3, 8, 9], [4, 6, 5, 7]),
        # Block 4: A2 B22 -> C21
        ([2, 3, 6, 7], [14, 15, 20, 21], [4, 6, 5, 7]),
        # Block 5: A1 B31 -> C31
        ([0, 1, 4, 5], [4, 5, 10, 11], [8, 10, 9, 11]),
        # Block 6: A2 B32 -> C31
        ([2, 3, 6, 7], [16, 17, 22, 23], [8, 10, 9, 11])
    ]
    
    col = 0
    for a_off, b_off, c_off in blocks:
        for i in range(7):
            for j in range(4):
                U[a_off[j], col] += U_strassen[i][j]
                V[b_off[j], col] += V_strassen[i][j]
                W[c_off[j], col] += W_strassen[i][j]
            col += 1
            
    return {"U": U, "V": V, "W": W}
