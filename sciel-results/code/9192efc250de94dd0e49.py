import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # 1. Construct 2x2x2 Strassen decomposition
    # n=2, m=2, p=2, rank=7
    n, m, p = 2, 2, 2
    rank_222 = 7
    U_222 = [[0] * rank_222 for _ in range(n * m)]
    V_222 = [[0] * rank_222 for _ in range(m * p)]
    W_222 = [[0] * rank_222 for _ in range(p * n)]
    
    # Define Strassen steps
    # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Indices are (i, j) for U, (j, k) for V, (k, i) for W
    # We convert to flat indices inside the loop
    
    steps = [
        # M1
        ([ (0,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1]),
        # M2
        ([ (1,0), (1,1) ], [1, 1], [ (0,0) ], [1], [ (0,1), (1,1) ], [1, -1]),
        # M3
        ([ (0,0) ], [1], [ (0,1), (1,1) ], [1, -1], [ (1,0), (1,1) ], [1, 1]),
        # M4
        ([ (1,1) ], [1], [ (1,0), (0,0) ], [1, -1], [ (0,0), (0,1) ], [1, 1]),
        # M5
        ([ (0,0), (1,0) ], [1, 1], [ (1,1) ], [1], [ (0,0), (1,0) ], [-1, 1]),
        # M6
        ([ (1,1), (0,0) ], [1, -1], [ (0,0), (0,1) ], [1, 1], [ (1,1) ], [1]),
        # M7
        ([ (0,1), (1,1) ], [1, -1], [ (1,0), (1,1) ], [1, 1], [ (0,0) ], [1])
    ]
    
    for r in range(rank_222):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = steps[r]
        for idx, val in zip(u_idx, u_val):
            i, j = idx
            U_222[i * m + j][r] = val
        for idx, val in zip(v_idx, v_val):
            j, k = idx
            V_222[j * p + k][r] = val
        for idx, val in zip(w_idx, w_val):
            k, i = idx
            W_222[k * n + i][r] = val

    # 2. Construct 2x2x4 decomposition
    # n=2, m=2, p=4, rank=14
    n2, m2, p2 = 2, 2, 4
    rank_224 = 14
    U_224 = [[0] * rank_224 for _ in range(n2 * m2)]
    V_224 = [[0] * rank_224 for _ in range(m2 * p2)]
    W_224 = [[0] * rank_224 for _ in range(p2 * n2)]
    
    for r in range(rank_222):
        # First block k in 0,1
        for i in range(n2):
            for j in range(m2):
                U_224[i * m2 + j][r] = U_222[i * 2 + j][r]
        
        for j in range(m2):
            for k in range(2): # k local 0,1
                V_224[j * p2 + k][r] = V_222[j * 2 + k][r]
        
        for k in range(2): # k local 0,1
            for i in range(n2):
                W_224[k * n2 + i][r] = W_222[k * 2 + i][r]
        
        # Second block k in 2,3
        r2 = r + rank_222
        for i in range(n2):
            for j in range(m2):
                U_224[i * m2 + j][r2] = U_222[i * 2 + j][r]
        
        for j in range(m2):
            for k in range(2): # k local 0,1 -> global 2,3
                V_224[j * p2 + (k + 2)][r2] = V_222[j * 2 + k][r]
        
        for k in range(2): # k local 0,1 -> global 2,3
            for i in range(n2):
                W_224[(k + 2) * n2 + i][r2] = W_222[k * 2 + i][r]

    # 3. Construct 2x4x4 decomposition
    # n=2, m=4, p=4, rank=28
    n3, m3, p3 = 2, 4, 4
    rank_244 = 28
    U_244 = [[0] * rank_244 for _ in range(n3 * m3)]
    V_244 = [[0] * rank_244 for _ in range(m3 * p3)]
    W_244 = [[0] * rank_244 for _ in range(p3 * n3)]
    
    for r in range(rank_224):
        # First block j in 0,1
        for i in range(n3):
            for j in range(2):
                U_244[i * m3 + j][r] = U_224[i * 2 + j][r]
        
        for j in range(2):
            for k in range(p3):
                V_244[j * p3 + k][r] = V_224[j * 4 + k][r]
        
        for k in range(p3):
            for i in range(n3):
                W_244[k * n3 + i][r] = W_224[k * 2 + i][r]
        
        # Second block j in 2,3
        r2 = r + rank_224
        for i in range(n3):
            for j in range(2): # j local 0,1 -> global 2,3
                U_244[i * m3 + (j + 2)][r2] = U_224[i * 2 + j][r]
        
        for j in range(2): # j local 0,1 -> global 2,3
            for k in range(p3):
                V_244[(j + 2) * p3 + k][r2] = V_224[j * 4 + k][r]
        
        for k in range(p3):
            for i in range(n3):
                W_244[k * n3 + i][r2] = W_224[k * 2 + i][r]

    return {"U": U_244, "V": V_244, "W": W_244}
