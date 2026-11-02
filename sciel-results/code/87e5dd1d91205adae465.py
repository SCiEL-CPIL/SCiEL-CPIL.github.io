def construct():
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 4, 8

    # 1. Define 2x2x2 Strassen coefficients
    # U2: 4x7, V2: 4x7, W2: 4x7
    # A index: i*2 + k
    # B index: k*2 + j
    # W index: j*2 + i (col*2 + row)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00(0), C01(2), C10(1), C11(3)
    W2[0, 0] = 1; W2[2, 0] = -1; W2[1, 0] = -1; W2[3, 0] = 1

    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10(1), C11(3)
    W2[1, 1] = 1; W2[3, 1] = 1

    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    # C01(2)
    W2[2, 2] = 1

    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    # C10(1), C11(3)
    W2[1, 3] = 1; W2[3, 3] = 1

    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C01(2), C11(3)
    W2[2, 4] = 1; W2[3, 4] = 1

    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    # C10(1)
    W2[1, 5] = 1

    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    # C01(2), C11(3)
    W2[2, 6] = 1; W2[3, 6] = 1

    # 2. Construct 4x4x4 (Rank 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # Block mapping:
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    # block_row = b // 2, block_col = b % 2

    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Iterate over all A elements
            for i_glob in range(4):
                for k_glob in range(4):
                    # Determine block and inner indices
                    i_block = i_glob // 2
                    k_block = k_glob // 2
                    u = i_glob % 2
                    v = k_glob % 2
                    block_idx = i_block * 2 + k_block # 0:00, 1:01, 2:10, 3:11
                    
                    # Coeff from outer s: U2[block_idx, s]
                    coeff_s = U2[block_idx, s]
                    if coeff_s == 0: continue
                    
                    # Coeff from inner t: U2[u*2 + v, t]
                    # Inner A index: u (row), v (col) -> u*2 + v
                    coeff_t = U2[u * 2 + v, t]
                    
                    U4[i_glob * 4 + k_glob, r] += coeff_s * coeff_t

            # Iterate over all B elements
            for k_glob in range(4):
                for j_glob in range(4):
                    k_block = k_glob // 2
                    j_block = j_glob // 2
                    k_inner = k_glob % 2
                    j_inner = j_glob % 2
                    block_idx = k_block * 2 + j_block # B block: row block, col block
                    
                    coeff_s = V2[block_idx, s]
                    if coeff_s == 0: continue
                    
                    # Inner B index: k_inner (row), j_inner (col) -> k_inner*2 + j_inner
                    coeff_t = V2[k_inner * 2 + j_inner, t]
                    
                    V4[k_glob * 4 + j_glob, r] += coeff_s * coeff_t

            # Iterate over all C elements
            for i_glob in range(4):
                for j_glob in range(4):
                    i_block = i_glob // 2
                    j_block = j_glob // 2
                    u = i_glob % 2
                    v = j_glob % 2
                    block_idx = i_block * 2 + j_block # C block: row block, col block
                    
                    coeff_s = W2[block_idx, s]
                    if coeff_s == 0: continue
                    
                    # Inner C index: v (col), u (row) -> v*2 + u
                    coeff_t = W2[v * 2 + u, t]
                    
                    # W index: j_glob * 4 + i_glob
                    W4[j_glob * 4 + i_glob, r] += coeff_s * coeff_t

    # 3. Extend to 4x4x8 (Rank 98)
    U8 = np.zeros((16, 98), dtype=int)
    V8 = np.zeros((32, 98), dtype=int)
    W8 = np.zeros((32, 98), dtype=int)

    # First 49 terms: B cols 0-3, C cols 0-3
    U8[:, :49] = U4
    
    # V8 first 49: B rows 0-3, cols 0-3
    # V4 index: k*4 + j (j in 0..3)
    # V8 index: k*8 + j (j in 0..3)
    for k in range(4):
        for j in range(4):
            V8[k * 8 + j, :49] = V4[k * 4 + j, :49]
    # V8 cols 4-7 are 0 for first 49

    # W8 first 49: C rows 0-3, cols 0-3
    # W4 index: j*4 + i (j in 0..3)
    # W8 index: j*4 + i (j in 0..3) -- Note: n=4 is fixed for W index formula k*n+i
    W8[:, :49] = W4

    # Next 49 terms: B cols 4-7, C cols 4-7
    U8[:, 49:] = U4
    
    # V8 next 49: B rows 0-3, cols 4-7
    # V4 index: k*4 + j (j in 0..3)
    # V8 index: k*8 + (j+4)
    for k in range(4):
        for j in range(4):
            V8[k * 8 + j + 4, 49:] = V4[k * 4 + j, 49:]
    # V8 cols 0-3 are 0 for next 49

    # W8 next 49: C rows 0-3, cols 4-7
    # W4 index: j*4 + i (j in 0..3)
    # W8 index: (j+4)*4 + i
    for j in range(4):
        for i in range(4):
            W8[(j + 4) * 4 + i, 49:] = W4[j * 4 + i, 49:]
    # W8 cols 0-3 are 0 for next 49

    return {
        "U": U8.tolist(),
        "V": V8.tolist(),
        "W": W8.tolist()
    }
