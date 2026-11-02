"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2, 3x3 blocks
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen decomposition for 2x2 matrices
    # U_local[u, v, s], V_local[v, w, s], W_local[w, u, s]
    # s in 0..6
    U_local = [[0]*2 for _ in range(2)] # u, v
    V_local = [[0]*2 for _ in range(2)] # v, w
    W_local = [[0]*2 for _ in range(2)] # w, u (output C_u,w)
    
    # We store them as lists of dicts or 3D lists
    # Let's use 3D lists: [s][u][v] etc
    U_loc = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    V_loc = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    W_loc = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0: m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    U_loc[0][0][0] = 1; U_loc[0][1][1] = 1
    V_loc[0][0][0] = 1; V_loc[0][1][1] = 1
    W_loc[0][0][0] = 1; W_loc[0][1][1] = 1
    
    # s=1: m2 = (c+d)e -> C10 += 1, C11 -= 1
    U_loc[1][1][0] = 1; U_loc[1][1][1] = 1
    V_loc[1][0][0] = 1
    W_loc[1][0][1] = 1; W_loc[1][1][1] = -1
    
    # s=2: m3 = a(f-h) -> C01 += 1, C11 += 1
    U_loc[2][0][0] = 1
    V_loc[2][1][0] = 1; V_loc[2][0][0] = -1
    W_loc[2][1][0] = 1; W_loc[2][1][1] = 1
    
    # s=3: m4 = d(g-e) -> C00 += 1, C10 += 1
    U_loc[3][1][1] = 1
    V_loc[3][0][1] = 1; V_loc[3][0][0] = -1
    W_loc[3][0][0] = 1; W_loc[3][0][1] = 1
    
    # s=4: m5 = (a+b)h -> C00 -= 1, C01 += 1
    U_loc[4][0][0] = 1; U_loc[4][0][1] = 1
    V_loc[4][1][1] = 1
    W_loc[4][0][0] = -1; W_loc[4][1][0] = 1
    
    # s=5: m6 = (c-a)(e+f) -> C11 += 1
    U_loc[5][1][0] = 1; U_loc[5][0][0] = -1
    V_loc[5][0][0] = 1; V_loc[5][0][1] = 1
    W_loc[5][1][1] = 1
    
    # s=6: m7 = (b-d)(g+h) -> C00 += 1
    U_loc[6][0][1] = 1; U_loc[6][1][1] = -1
    V_loc[6][1][0] = 1; V_loc[6][1][1] = 1
    W_loc[6][0][0] = 1
    
    rank = (num_blocks * num_blocks) * num_blocks * 7 # 9 * 3 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for I in range(num_blocks):
        for K in range(num_blocks):
            for J in range(num_blocks):
                for s in range(7):
                    for u in range(block_size):
                        for v in range(block_size):
                            i = I * block_size + u
                            j = J * block_size + v
                            # U[i*m + j, r]
                            U[i * m + j][r] = U_loc[s][u][v]
                            
                            for w in range(block_size):
                                k = K * block_size + w
                                # V[j*p + k, r]
                                V[j * p + k][r] = V_loc[s][v][w]
                                # W[k*n + i, r]
                                W[k * n + i][r] = W_loc[s][w][u]
                    r += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
