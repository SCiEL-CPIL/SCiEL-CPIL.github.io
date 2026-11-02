"""Matrix multiplication 2x4 x 4x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  -> c = k*2 + i
    #    c=0: k=0,i=0
    #    c=1: k=0,i=1
    #    c=2: k=1,i=0
    #    c=3: k=1,i=1
    
    # u[t][a_idx], v[t][b_idx], w[t][c_idx]
    u_terms = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, -1]  # M6
    ]
    
    v_terms = [
        [1, 0, 0, 1],  # M0
        [1, 0, 0, 0],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, -1], # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1]   # M6
    ]
    
    w_terms = [
        [1, 0, 0, 1],  # M0
        [0, 1, 0, -1], # M1
        [0, 0, 1, 1],  # M2
        [1, 1, 0, 0],  # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 0, 0]   # M6
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Blocks: (j_offset, k_offset)
    # A is split into A_L (j 0,1) and A_R (j 2,3)
    # B is split into 4 blocks
    # C_L (k 0,1) and C_R (k 2,3)
    blocks = [
        (0, 0), # A_L * B_LL -> C_L
        (2, 0), # A_R * B_RL -> C_L
        (0, 2), # A_L * B_LR -> C_R
        (2, 2)  # A_R * B_RR -> C_R
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for t in range(7):
            # Add term t to columns starting at col
            # u
            for a_local in range(4):
                coeff = u_terms[t][a_local]
                if coeff != 0:
                    # a_local = i' * 2 + j'
                    i_prime = a_local // 2
                    j_prime = a_local % 2
                    i = i_prime # i_off is 0
                    j = j_prime + j_off
                    a_global = i * m + j
                    U[a_global][col] += coeff
            
            # v
            for b_local in range(4):
                coeff = v_terms[t][b_local]
                if coeff != 0:
                    # b_local = j' * 2 + k'
                    j_prime = b_local // 2
                    k_prime = b_local % 2
                    j = j_prime + j_off
                    k = k_prime + k_off
                    b_global = j * p + k
                    V[b_global][col] += coeff
            
            # w
            for c_local in range(4):
                coeff = w_terms[t][c_local]
                if coeff != 0:
                    # c_local = k' * 2 + i'
                    k_prime = c_local // 2
                    i_prime = c_local % 2
                    k = k_prime + k_off
                    i = i_prime
                    c_global = k * n + i
                    W[c_global][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
