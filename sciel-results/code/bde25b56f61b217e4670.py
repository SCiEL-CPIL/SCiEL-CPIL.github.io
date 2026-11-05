"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0, 2, 4]
    
    total_rank = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                si, sj, sk = block_sizes[I], block_sizes[J], block_sizes[K]
                if si == 2 and sj == 2 and sk == 2:
                    total_rank += 7
                else:
                    total_rank += si * sj * sk
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                si, sj, sk = block_sizes[I], block_sizes[J], block_sizes[K]
                i_off, j_off, k_off = offsets[I], offsets[J], offsets[K]
                
                if si == 2 and sj == 2 and sk == 2:
                    U_loc = [[1,0,1,0,1,-1,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0],[1,1,0,1,0,0,-1]]
                    V_loc = [[1,1,0,0,0,1,0],[0,0,1,0,0,1,0],[0,0,0,1,0,0,1],[0,0,-1,-1,1,0,1]]
                    W_loc = [[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,-1,0,1,0,0,0],[1,0,1,0,0,1,0]]
                    for r in range(7):
                        for ip in range(2):
                            for jp in range(2):
                                u = U_loc[ip*2+jp][r]
                                if u:
                                    U[(i_off+ip)*m + (j_off+jp)][col+r] = u
                        for jp in range(2):
                            for kp in range(2):
                                v = V_loc[jp*2+kp][r]
                                if v:
                                    V[(j_off+jp)*p + (k_off+kp)][col+r] = v
                        for kp in range(2):
                            for ip in range(2):
                                w = W_loc[kp*2+ip][r]
                                if w:
                                    W[(k_off+kp)*n + (i_off+ip)][col+r] = w
                    col += 7
                else:
                    for ip in range(si):
                        for jp in range(sj):
                            for kp in range(sk):
                                U[(i_off+ip)*m + (j_off+jp)][col] = 1
                                V[(j_off+jp)*p + (k_off+kp)][col] = 1
                                W[(k_off+kp)*n + (i_off+ip)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
