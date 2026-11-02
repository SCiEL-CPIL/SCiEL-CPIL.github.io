"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 blocks of Strassen 2x2x2
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_S[i', j', c]
    U_S = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    # V_S[j', k', c]
    V_S = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    # W_S[k', i', c]
    W_S = [
        [(0,0,1), (1,1,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1), (0,1,1)],
        [(0,0,-1), (1,0,1)],
        [(1,1,1)],
        [(0,0,1), (1,0,-1)]
    ]
    
    col = 0
    for k2 in range(2):
        for j2 in range(2):
            for k1b in range(2):
                for c_s in range(7):
                    # U updates
                    for i_p, j_p, val in U_S[c_s]:
                        i = i_p
                        j = 2 * j2 + j_p
                        U[i * m + j][col] += val
                    # V updates
                    for j_p, k_p, val in V_S[c_s]:
                        j = 2 * j2 + j_p
                        k = 4 * k2 + 2 * k1b + k_p
                        V[j * p + k][col] += val
                    # W updates
                    for k_p, i_p, val in W_S[c_s]:
                        k = 4 * k2 + 2 * k1b + k_p
                        i = i_p
                        W[k * n + i][col] += val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
