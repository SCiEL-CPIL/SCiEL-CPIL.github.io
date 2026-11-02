import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 49 for 4x4 matrix multiplication
    # Using Kronecker product of two 2x2 Strassen decompositions
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # Indices 0..3 correspond to 00, 01, 10, 11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 1, 0, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Fill U
    # x = r * 4 + k
    for x in range(16):
        r = x // 4
        k = x % 4
        br = r // 2
        lr = r % 2
        bk = k // 2
        lk = k % 2
        
        idx_out = br * 2 + bk
        idx_in = lr * 2 + lk
        
        for s in range(7):
            u_out = U2[idx_out][s]
            if u_out == 0: continue
            for t in range(7):
                u_in = U2[idx_in][t]
                if u_in == 0: continue
                R = 7 * s + t
                U[x][R] = u_out * u_in

    # Fill V
    # y = k * 4 + c
    for y in range(16):
        k = y // 4
        c = y % 4
        bk = k // 2
        lk = k % 2
        bc = c // 2
        lc = c % 2
        
        idx_out = bk * 2 + bc
        idx_in = lk * 2 + lc
        
        for s in range(7):
            v_out = V2[idx_out][s]
            if v_out == 0: continue
            for t in range(7):
                v_in = V2[idx_in][t]
                if v_in == 0: continue
                R = 7 * s + t
                V[y][R] = v_out * v_in

    # Fill W
    # z = r * 4 + c (assuming row-major for C, matching W[k*n+i] with k=r, i=c)
    for z in range(16):
        r = z // 4
        c = z % 4
        br = r // 2
        lr = r % 2
        bc = c // 2
        lc = c % 2
        
        idx_out = br * 2 + bc
        idx_in = lr * 2 + lc
        
        for s in range(7):
            w_out = W2[idx_out][s]
            if w_out == 0: continue
            for t in range(7):
                w_in = W2[idx_in][t]
                if w_in == 0: continue
                R = 7 * s + t
                W[z][R] = w_out * w_in

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
