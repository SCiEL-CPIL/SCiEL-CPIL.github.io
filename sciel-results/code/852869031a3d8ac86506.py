import numpy as np

def construct():
    # 2x2 Strassen vectors
    # Flattened as [row0col0, row0col1, row1col0, row1col1]
    U2 = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    V2 = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    W2 = [
        [1, 0, 0, 1],  # P1: C00, C11
        [0, 0, 1, 0],  # P2: C10
        [0, 1, 0, -1], # P3: C01, -C11
        [1, 0, 1, 0],  # P4: C00, C10
        [0, 1, 0, 1],  # P5: C01, C11
        [-1, 0, 0, 0], # P6: -C00
        [1, 0, 0, -1]  # P7: C00, -C11
    ]
    
    rank = 49
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U4 = [[0] * rank for _ in range(16)]
    V4 = [[0] * rank for _ in range(16)]
    W4 = [[0] * rank for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            R = r * 7 + s
            # Iterate over 4x4 indices
            # i = 2*i1 + i2, j = 2*j1 + j2
            # U4[i*4+j, R] = U2[r][i1*2+j1] * U2[s][i2*2+j2]
            # Note: U2 index is i*2 + j for 2x2 matrix flattened row-major.
            # So U2[r][idx] where idx = i*2 + j.
            
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    
                    # U part
                    u_val = U2[r][i1 * 2 + j1] * U2[s][i2 * 2 + j2]
                    U4[i * 4 + j][R] = u_val
                    
                    # V part
                    # V4[j*4+k, R] = V2[r][j1*2+k1] * V2[s][j2*2+k2]
                    # We can fill V4 in the same loop if we iterate k, but V depends on j and k.
                    # Let's fill V4 and W4 in separate loops or combined.
                    
            # Fill V4
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    v_val = V2[r][j1 * 2 + k1] * V2[s][j2 * 2 + k2]
                    V4[j * 4 + k][R] = v_val
                    
            # Fill W4
            # W4[k*n+i, R] is coeff for C[i,k]
            # Formula: W2[r][i1*2+k1] * W2[s][i2*2+k2]
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    w_val = W2[r][i1 * 2 + k1] * W2[s][i2 * 2 + k2]
                    W4[k * 4 + i][R] = w_val

    return {"U": U4, "V": V4, "W": W4}
