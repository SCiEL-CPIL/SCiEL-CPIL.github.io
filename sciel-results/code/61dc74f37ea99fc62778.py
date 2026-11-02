"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U_local, V_local, W_local are 2x2 matrices (lists of lists)
    # Indices 0..1
    strassen_U = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 0], [-1, 1]] # M7
    ]
    strassen_V = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [1, 0]]  # M7
    ]
    strassen_W = [
        [[1, 0], [0, -1]], # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [1, 0]],  # M3
        [[-1, 1], [0, 0]], # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, -1]], # M6
        [[-1, 1], [0, 0]]  # M7
    ]
    
    col = 0
    # 6 instances of Strassen
    # 3 for A1 (j=0,1) and B1 (j=0,1, k=0..5)
    # 3 for A2 (j=2,3) and B2 (j=2,3, k=0..5)
    
    for t in range(6):
        is_top = (t < 3)
        sub_t = t if is_top else t - 3
        
        # Global offsets
        # A rows: i in 0..1
        # A cols: j in 0..1 (top) or 2..3 (bottom)
        j_offset_A = 0 if is_top else 2
        
        # B rows: j in 0..1 (top) or 2..3 (bottom)
        j_offset_B = 0 if is_top else 2
        
        # B cols: k in 0..5. Sub-block sub_t occupies k in 2*sub_t .. 2*sub_t+1
        k_offset_B = 2 * sub_t
        
        for s in range(7):
            u_loc = strassen_U[s]
            v_loc = strassen_V[s]
            w_loc = strassen_W[s]
            
            for i in range(2):
                for j_l in range(2):
                    val_u = u_loc[i][j_l]
                    if val_u != 0:
                        j_global = j_offset_A + j_l
                        U[i * m + j_global][col] += val_u
                        
            for j_l in range(2):
                for k_l in range(2):
                    val_v = v_loc[j_l][k_l]
                    if val_v != 0:
                        j_global = j_offset_B + j_l
                        k_global = k_offset_B + k_l
                        V[j_global * p + k_global][col] += val_v
                        
            for k_l in range(2):
                for i in range(2):
                    val_w = w_loc[k_l][i]
                    if val_w != 0:
                        k_global = k_offset_B + k_l
                        W[k_global * n + i][col] += val_w
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
