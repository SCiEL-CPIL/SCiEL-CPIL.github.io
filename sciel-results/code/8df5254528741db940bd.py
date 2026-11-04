"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Indices for U2: row is i*2+j (0..3), col is mu (0..6)
    # i, j in {0, 1}
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, -1],  # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ]
    
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, -1, 1, 0, 1]  # b11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ]
    
    n, m, p = DIMENSIONS
    # n=m=p=4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker product construction
    # i = 2*i1 + i2, j = 2*j1 + j2
    # k = 2*k1 + k2
    # col = mu*7 + nu
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    # U[i, j]
                    row_u = i * m + j
                    for mu in range(7):
                        u1 = U2[i1 * 2 + j1][mu]
                        if u1 == 0: continue
                        for nu in range(7):
                            u2 = U2[i2 * 2 + j2][nu]
                            if u2 == 0: continue
                            val = u1 * u2
                            col = mu * 7 + nu
                            U[row_u][col] = val
                            
    for j1 in range(2):
        for j2 in range(2):
            j = 2 * j1 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    # V[j, k]
                    row_v = j * p + k
                    for mu in range(7):
                        v1 = V2[j1 * 2 + k1][mu]
                        if v1 == 0: continue
                        for nu in range(7):
                            v2 = V2[j2 * 2 + k2][nu]
                            if v2 == 0: continue
                            val = v1 * v2
                            col = mu * 7 + nu
                            V[row_v][col] = val
                            
    for k1 in range(2):
        for k2 in range(2):
            k = 2 * k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    # W[k, i]
                    row_w = k * n + i
                    for mu in range(7):
                        w1 = W2[k1 * 2 + i1][mu]
                        if w1 == 0: continue
                        for nu in range(7):
                            w2 = W2[k2 * 2 + i2][nu]
                            if w2 == 0: continue
                            val = w1 * w2
                            col = mu * 7 + nu
                            W[row_w][col] = val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
