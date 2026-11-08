import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Local 3x2x2 decomposition (Rank 11)
    U_loc = np.zeros((6, 11), dtype=np.float64)
    V_loc = np.zeros((4, 11), dtype=np.float64)
    W_loc = np.zeros((6, 11), dtype=np.float64)
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # 0: A(:, 0:1) * B(0:1, 0:1) -> C(:, 0:1)
    col_offset = 0
    for r in range(3):
        for c in range(2):
            U[(r) * m + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(r) * p + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 1: A(:, 2:3) * B(2:3, 0:1) -> C(:, 0:1)
    col_offset = 11
    for r in range(3):
        for c in range(2):
            U[(r) * m + 2 + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(2 + r) * p + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 2: A(:, 0:1) * B(0:1, 2:3) -> C(:, 2:3)
    col_offset = 22
    for r in range(3):
        for c in range(2):
            U[(r) * m + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(r) * p + 2 + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(2 + c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 3: A(:, 2:3) * B(2:3, 2:3) -> C(:, 2:3)
    col_offset = 33
    for r in range(3):
        for c in range(2):
            U[(r) * m + 2 + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(2 + r) * p + 2 + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(2 + c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
