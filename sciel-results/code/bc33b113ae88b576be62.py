import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (2x2) * B_1 (2x8)
    for k in range(4):
        e_idx = 2 * k
        f_idx = 2 * k + 1
        g_idx = 8 + 2 * k
        h_idx = 8 + 2 * k + 1
        
        c00_w = 4 * k
        c01_w = 4 * k + 2
        c10_w = 4 * k + 1
        c11_w = 4 * k + 3
        
        # m1
        U[0][col] = 1
        U[4][col] = 1
        V[e_idx][col] = 1
        V[h_idx][col] = 1
        W[c00_w][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m2
        U[3][col] = 1
        U[4][col] = 1
        V[e_idx][col] = 1
        W[c10_w][col] = 1
        W[c11_w][col] = -1
        col += 1
        
        # m3
        U[0][col] = 1
        V[f_idx][col] = 1
        V[h_idx][col] = -1
        W[c01_w][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m4
        U[4][col] = 1
        V[g_idx][col] = 1
        V[f_idx][col] = -1
        W[c00_w][col] = 1
        W[c10_w][col] = 1
        col += 1
        
        # m5
        U[0][col] = 1
        U[1][col] = 1
        V[h_idx][col] = 1
        W[c00_w][col] = -1
        W[c01_w][col] = 1
        col += 1
        
        # m6
        U[3][col] = 1
        U[4][col] = -1
        V[e_idx][col] = 1
        V[f_idx][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m7
        U[1][col] = 1
        U[4][col] = -1
        V[g_idx][col] = 1
        V[h_idx][col] = 1
        W[c00_w][col] = 1
        col += 1

    # Part 2: A_2 (2x1) * B_2 (1x8)
    for i in range(2):
        a_idx = 2 + i
        for k in range(8):
            b_idx = 16 + k
            c_idx = 2 * k + i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
