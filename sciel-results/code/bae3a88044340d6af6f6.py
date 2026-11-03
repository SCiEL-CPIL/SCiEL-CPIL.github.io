import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen rank-7 decomposition for 2x2x2
    U2 = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    V2 = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    W2 = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    rank = 49
    U = [[0] * rank for _ in range(n * m)] # 16 x 49
    V = [[0] * rank for _ in range(m * p)] # 32 x 49 (m=4, p=8 -> 32)
    W = [[0] * rank for _ in range(p * n)] # 32 x 49 (p=8, n=4 -> 32)
    
    def idx2(i, j):
        return i * 2 + j
        
    for s in range(7):
        for t in range(7):
            col_idx = s * 7 + t
            
            # U4: 4x4, flattened to 16
            U4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    r = I // 2
                    c = J // 2
                    i_loc = I % 2
                    j_loc = J % 2
                    coeff = U2[s][idx2(r, c)] * U2[t][idx2(i_loc, j_loc)]
                    U4[I * 4 + J] = coeff
                    
            # V4: 4x4, flattened to 16
            V4 = [0] * 16
            for J in range(4):
                for K in range(4):
                    r = J // 2
                    c = K // 2
                    j_loc = J % 2
                    k_loc = K % 2
                    coeff = V2[s][idx2(r, c)] * V2[t][idx2(j_loc, k_loc)]
                    V4[J * 4 + K] = coeff
                    
            # W4: 4x4, flattened to 16
            W4 = [0] * 16
            for I in range(4):
                for K in range(4):
                    r = I // 2
                    c = K // 2
                    i_loc = I % 2
                    k_loc = K % 2
                    coeff = W2[s][idx2(r, c)] * W2[t][idx2(i_loc, k_loc)]
                    W4[I * 4 + K] = coeff
            
            # Fill U
            for i in range(16):
                U[i][col_idx] = U4[i]
                
            # Fill V (m=4, p=8)
            # V has shape (4*8, 49) = (32, 49)
            # V[j, k] for j in 0..3, k in 0..7
            # We use V4 for k in 0..3 and duplicate for k in 4..7
            for j in range(4):
                for k in range(4):
                    val = V4[j * 4 + k]
                    V[j * 8 + k][col_idx] = val
                    V[j * 8 + (k + 4)][col_idx] = val
                    
            # Fill W (p=8, n=4)
            # W has shape (8*4, 49) = (32, 49)
            # W[k, i] for k in 0..7, i in 0..3
            # W4[I, K] corresponds to C[I, K]
            # W index is k * n + i = k * 4 + i
            for k in range(4):
                for i in range(4):
                    val = W4[i * 4 + k]
                    W[k * 4 + i][col_idx] = val
                    W[(k + 4) * 4 + i][col_idx] = val

    return {"U": U, "V": V, "W": W}
