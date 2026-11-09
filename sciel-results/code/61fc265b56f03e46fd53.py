def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # 2x2 Strassen vectors
    # u, v, w are length 4
    s2_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [1, 0, -1, 0], # M6 (A10-A00) -> [0,0,1,0] - [1,0,0,0] = [-1,0,1,0]
        [0, 1, 0, -1]  # M7 (A01-A11) -> [0,1,0,0] - [0,0,0,1] = [0,1,0,-1]
    ]
    # Correction:
    # M6: A10 - A00. A10 is index 2, A00 is index 0.
    # s2_u[5] should be [-1, 0, 1, 0]
    # M7: A01 - A11. A01 is index 1, A11 is index 3.
    # s2_u[6] should be [0, 1, 0, -1]
    
    s2_u = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]
    ]
    
    s2_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    s2_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [0, 0, 1, 1], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, -1]  # M7
    ]
    
    # High level Strassen terms
    # A_k = sum c_ijk A_jk
    # B_k = sum d_ijk B_jk
    # C_jk += e_kjk M_k
    
    # Indices: 00, 01, 10, 11
    # A blocks: 0, 1, 2, 3
    # B blocks: 0, 1, 2, 3
    
    # M1: (A00+A11)(B00+B11)
    # A: 1,0,0,1 -> B: 1,0,0,1
    # C: 1,0,0,1 (C00 += 1, C11 += 1)
    
    # M2: (A10+A11)B00
    # A: 0,0,1,1 -> B: 1,0,0,0
    # C: 0,0,1,0 (C10 += 1)
    
    # M3: A00(B01-B11)
    # A: 1,0,0,0 -> B: 0,1,0,-1
    # C: 0,1,0,0 (C01 += 1)
    
    # M4: A11(B10-B00)
    # A: 0,0,0,1 -> B: 0,0,1,-1
    # C: 0,0,1,1 (C10 += 1, C11 += 1)
    
    # M5: (A00+A01)B11
    # A: 1,1,0,0 -> B: 0,0,0,1
    # C: -1,1,0,0 (C00 -= 1, C01 += 1)
    
    # M6: (A10-A00)(B00+B01)
    # A: -1,0,1,0 -> B: 1,1,0,0
    # C: 0,0,0,1 (C11 += 1)
    
    # M7: (A01-A11)(B10+B11)
    # A: 0,1,0,-1 -> B: 0,0,1,1
    # C: 1,0,0,-1 (C00 += 1, C11 -= 1)
    
    high_a = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    high_b = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [0,0,1,-1],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    high_c = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,1,1],
        [-1,1,0,0],
        [0,0,0,1],
        [1,0,0,-1]
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            # Combine high level k and low level l
            # U vector: sum over blocks j of high_a[k][j] * s2_u[l]
            # V vector: sum over blocks j of high_b[k][j] * s2_v[l]
            # W vector: sum over blocks j of high_c[k][j] * s2_w[l]
            
            u_vec = [0]*16
            v_vec = [0]*16
            w_vec = [0]*16
            
            for j in range(4):
                if high_a[k][j] != 0:
                    for ii in range(4):
                        u_vec[j*4 + ii] += high_a[k][j] * s2_u[l][ii]
                if high_b[k][j] != 0:
                    for ii in range(4):
                        v_vec[j*4 + ii] += high_b[k][j] * s2_v[l][ii]
                if high_c[k][j] != 0:
                    for ii in range(4):
                        w_vec[j*4 + ii] += high_c[k][j] * s2_w[l][ii]
            
            # Add to U, V, W for first 4x4x4 block (k=0..3)
            for i in range(16):
                U[i][col] = u_vec[i]
            for i in range(16):
                V[i][col] = v_vec[i]
            for i in range(16):
                W[i][col] = w_vec[i]
                
            col += 1
            
            # Add to U, V, W for second 4x4x4 block (k=4..7)
            # V and W indices shift by 16 in the k dimension?
            # V index is j*p + k. p=8.
            # For first block, k=0..3.
            # For second block, k=4..7.
            # So V index shifts by 4? No, V index is j*8 + k.
            # If we use the same V vector, it corresponds to k=0..3.
            # For the second block, we need to shift the k index by 4.
            # So V[i][col] should be non-zero at j*8 + (k+4).
            # Similarly W index is k*n + i = k*4 + i.
            # Shift by 4*4 = 16.
            
            for i in range(16):
                U[i][col] = u_vec[i]
            
            # Shift V
            for i in range(16):
                if v_vec[i] != 0:
                    j = i // 4
                    kk = i % 4
                    new_i = j * 8 + (kk + 4)
                    V[new_i][col] = v_vec[i]
                    
            # Shift W
            for i in range(16):
                if w_vec[i] != 0:
                    kk = i // 4
                    ii = i % 4
                    new_i = (kk + 4) * 4 + ii
                    W[new_i][col] = w_vec[i]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
