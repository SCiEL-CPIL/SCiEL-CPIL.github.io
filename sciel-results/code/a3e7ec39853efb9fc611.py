"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks of Strassen (2x2x2), each rank 7. Total 56.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: [a00, a01, a10, a11]
    # v: [b00, b01, b10, b11]
    # w: [c00, c01, c10, c11]
    strassen_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1],# P6
        [0, 1, 0, -1] # P7
    ]
    strassen_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    strassen_w = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [0, 1, 0, -1],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # Iterate over 8 sub-problems: 2 halves of A/B (h), 4 quarters of output (q)
    for h in range(2):
        for q in range(4):
            base_col = 7 * (h * 4 + q)
            
            # Local indices mapping to global indices
            # U (n*m = 8): i*m + j
            u_idx = [
                0 * m + (h * 2 + 0), # a00
                0 * m + (h * 2 + 1), # a01
                1 * m + (h * 2 + 0), # a10
                1 * m + (h * 2 + 1)  # a11
            ]
            
            # V (m*p = 32): j*p + k
            v_idx = [
                (h * 2 + 0) * p + (2 * q + 0), # b00
                (h * 2 + 0) * p + (2 * q + 1), # b01
                (h * 2 + 1) * p + (2 * q + 0), # b10
                (h * 2 + 1) * p + (2 * q + 1)  # b11
            ]
            
            # W (p*n = 16): k*n + i
            w_idx = [
                (2 * q + 0) * n + 0, # c00
                (2 * q + 1) * n + 0, # c01
                (2 * q + 0) * n + 1, # c10
                (2 * q + 1) * n + 1  # c11
            ]
            
            for r in range(7):
                col = base_col + r
                # Update U
                for local_i in range(4):
                    if strassen_u[r][local_i] != 0:
                        U[u_idx[local_i]][col] += strassen_u[r][local_i]
                # Update V
                for local_j in range(4):
                    if strassen_v[r][local_j] != 0:
                        V[v_idx[local_j]][col] += strassen_v[r][local_j]
                # Update W
                for local_k in range(4):
                    if strassen_w[r][local_k] != 0:
                        W[w_idx[local_k]][col] += strassen_w[r][local_k]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
