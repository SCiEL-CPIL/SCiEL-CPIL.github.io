def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    strassen_2x2 = [
        ([1,0,0,1], [1,1,0,0], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,1,0], [1,0,1,0]),
        ([1,0,1,0], [0,0,0,1], [-1,0,0,1]),
        ([0,1,0,1], [1,0,-1,0], [0,0,0,1]),
        ([1,0,-1,0], [0,1,0,1], [1,0,0,0]),
    ]
    
    # Level 1 Strassen for 4x4x4
    # U1[i_block, j_block] for each of 7 terms
    # V1[j_block, k_block] for each of 7 terms  
    # W1[k_block, i_block] for each of 7 terms
    
    level1_U = [
        [1,0,0,1],  # M1: A00+A11
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,0,1,0],  # M5: A00+A10
        [0,1,0,1],  # M6: A01+A11
        [1,0,-1,0], # M7: A00-A10
    ]
    level1_V = [
        [1,1,0,0],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,0,1,-1], # M3: B10-B11
        [0,-1,1,0], # M4: B01-B10
        [0,0,0,1],  # M5: B11
        [1,0,-1,0], # M6: B00-B10
        [0,1,0,1],  # M7: B01+B11
    ]
    level1_W = [
        [1,0,0,1],  # M1: C00+C11
        [0,0,1,-1], # M2: C10-C11
        [0,1,0,1],  # M3: C01+C11
        [1,0,1,0],  # M4: C00+C10
        [-1,0,0,1], # M5: -C00+C11
        [0,0,0,1],  # M6: C11
        [1,0,0,0],  # M7: C00
    ]
    
    # For each of the two 4x4x4 blocks (C1 and C2)
    for block in range(2):
        base = block * 49
        for t in range(7):  # level 1 term
            for s in range(7):  # level 2 term
                r = base + t*7 + s
                U2, V2, W2 = strassen_2x2[s]
                
                # U: 16 entries (4x4), indexed by i*4+j
                for i in range(4):
                    for j in range(4):
                        ib, jb = i//2, j//2
                        il, jl = i%2, j%2
                        U[i*4+j][r] = level1_U[t][ib*2+jb] * U2[il*2+jl]
                
                # V: 32 entries (4x8), indexed by j*8+k
                for j in range(4):
                    for k in range(4):
                        jb, kb = j//2, k//2
                        jl, kl = j%2, k%2
                        V[j*8 + (block*4 + k)][r] = level1_V[t][jb*2+kb] * V2[jl*2+kl]
                
                # W: 32 entries (8x4), indexed by k*4+i
                for k in range(4):
                    for i in range(4):
                        kb, ib = k//2, i//2
                        kl, il = k%2, i%2
                        W[(block*4 + k)*4 + i][r] = level1_W[t][kb*2+ib] * W2[kl*2+il]
    
    return {"U": U, "V": V, "W": W}
