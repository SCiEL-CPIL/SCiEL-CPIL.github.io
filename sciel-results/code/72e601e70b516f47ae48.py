"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 multiplication
    # u[i, j] -> flattened i*2+j
    # v[j, k] -> flattened j*2+k
    # w[k, i] -> flattened k*2+i
    
    u = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [-1, 0, 1, 0],# 6
        [0, 1, 0, -1] # 7
    ]
    
    v = [
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 0, 1, -1],# 3
        [-1, 1, 0, 0],# 4
        [0, 0, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ]
    
    w = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, -1],# 2
        [0, 0, 1, 1], # 3
        [1, 0, 1, 0], # 4
        [-1, 0, 1, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 0, 0, 0]  # 7
    ]
    
    # Iterate over 49 combinations
    col = 0
    for s in range(7):
        for t in range(7):
            # Construct U column
            # U index: i*4+j, i=2*i1+i2, j=2*j1+j2
            # u_s[i1, j1] * u_t[i2, j2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            idx_u = i * 4 + j
                            val = u[s][i1 * 2 + j1] * u[t][i2 * 2 + j2]
                            if val != 0:
                                U[idx_u][col] = val
                                
            # Construct V column
            # V index: j*4+k, j=2*j1+j2, k=2*k1+k2
            # v_s[j1, k1] * v_t[j2, k2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            idx_v = j * 4 + k
                            val = v[s][j1 * 2 + k1] * v[t][j2 * 2 + k2]
                            if val != 0:
                                V[idx_v][col] = val
                                
            # Construct W column
            # W index: k*4+i, k=2*k1+k2, i=2*i1+i2
            # w_s[k1, i1] * w_t[k2, i2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            idx_w = k * 4 + i
                            val = w[s][k1 * 2 + i1] * w[t][k2 * 2 + i2]
                            if val != 0:
                                W[idx_w][col] = val
                                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
