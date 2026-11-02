import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B local indices: 0:B00, 1:B01, 2:B10, 3:B11
    # C local indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, -1],# M6
        [1, -1, 0, 0] # M7
    ]
    
    # Block 1: k in {0, 1}
    for t in range(7):
        col = t
        for i in range(4):
            U[i][col] = A_coeffs[t][i]
        v_map = [0, 1, 4, 5]
        for local_j, val in enumerate(B_coeffs[t]):
            if val != 0:
                V[v_map[local_j]][col] = val
        w_map = [0, 1, 2, 3]
        for local_k, val in enumerate(C_coeffs[t]):
            if val != 0:
                W[w_map[local_k]][col] = val

    # Block 2: k in {2, 3}
    for t in range(7):
        col = t + 7
        for i in range(4):
            U[i][col] = A_coeffs[t][i]
        v_map = [2, 3, 6, 7]
        for local_j, val in enumerate(B_coeffs[t]):
            if val != 0:
                V[v_map[local_j]][col] = val
        w_map = [4, 5, 6, 7]
        for local_k, val in enumerate(C_coeffs[t]):
            if val != 0:
                W[w_map[local_k]][col] = val

    return {"U": U, "V": V, "W": W}
