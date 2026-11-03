"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 Strassen components = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen vectors for 2x2 matrix multiplication
    # Indices: i, j, k in {0, 1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    strassen_u = [
        [1, 0, 0, 0], # P1: X00
        [0, 0, 1, 0], # P2: X10
        [1, 0, 1, 0], # P3: X00 + X10
        [0, 1, 0, 1], # P4: X01 + X11
        [0, 1, 0, 0], # P5: X01
        [1, 0, 0, -1],# P6: X00 - X11
        [0, 1, -1, 0] # P7: X01 - X10
    ]
    strassen_v = [
        [1, 0, 0, 1], # P1: Z00 + Z11
        [1, 1, 0, 0], # P2: Z00 + Z01
        [0, 0, 0, 1], # P3: Z11
        [1, 0, 0, 0], # P4: Z00
        [0, 0, 1, 1], # P5: Z10 + Z11
        [0, 1, 0, 1], # P6: Z01 + Z11
        [1, 0, -1, 0] # P7: Z00 - Z10
    ]
    strassen_w = [
        [1, 0, 0, 1], # P1: Y00 + Y11
        [0, 1, 0, -1],# P2: Y10 - Y11
        [-1, 1, 0, 0],# P3: -Y00 + Y10
        [0, 0, 1, -1],# P4: Y01 - Y11
        [0, 0, 1, 0], # P5: Y01
        [0, 1, 0, 1], # P6: Y10 + Y11
        [1, 0, 0, 0]  # P7: Y00
    ]
    
    # Global indices:
    # i in {0, 1}
    # j in {0, 1, 2, 3} -> j = 2*j1 + j2
    # k in {0..7} -> k = 2*k1 + k2
    
    col = 0
    for j1 in range(2):
        for k1 in range(4):
            for s in range(7):
                # Map local indices to global indices
                # Local i -> Global i
                # Local j -> Global j2 (part of j)
                # Local k -> Global k2 (part of k)
                
                for i in range(2):
                    for j2 in range(2):
                        local_u_idx = i * 2 + j2
                        val_u = strassen_u[s][local_u_idx]
                        if val_u != 0:
                            global_j = 2 * j1 + j2
                            global_u_idx = i * m + global_j
                            U[global_u_idx][col] += val_u
                            
                for j2 in range(2):
                    for k2 in range(2):
                        local_v_idx = j2 * 2 + k2
                        val_v = strassen_v[s][local_v_idx]
                        if val_v != 0:
                            global_j = 2 * j1 + j2
                            global_k = 2 * k1 + k2
                            global_v_idx = global_j * p + global_k
                            V[global_v_idx][col] += val_v
                            
                for k2 in range(2):
                    for i in range(2):
                        local_w_idx = k2 * 2 + i
                        val_w = strassen_w[s][local_w_idx]
                        if val_w != 0:
                            global_k = 2 * k1 + k2
                            global_w_idx = global_k * n + i
                            W[global_w_idx][col] += val_w
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
