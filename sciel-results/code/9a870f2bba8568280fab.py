"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Local indices: 0:00, 1:01, 2:10, 3:11
    # U, V, W vectors for 7 steps
    steps = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # u: A block index (0, 1)
    # v: B block index (0, 1, 2)
    for u in range(2):
        for v in range(3):
            for U_loc, V_loc, W_loc in steps:
                # Map local indices to global
                # Local A (i', j') -> Global (i', 2u+j')
                # Local B (j'', k') -> Global (2u+j'', 2v+k')
                # Local C (i', k') -> Global (i', 2v+k')
                
                for li in range(2):
                    for lj in range(2):
                        l_idx = li * 2 + lj
                        val = U_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_j = 2 * u + lj
                            U[g_i * m + g_j][col] += val
                
                for lj in range(2):
                    for lk in range(2):
                        l_idx = lj * 2 + lk
                        val = V_loc[l_idx]
                        if val != 0:
                            g_j = 2 * u + lj
                            g_k = 2 * v + lk
                            V[g_j * p + g_k][col] += val
                
                for li in range(2):
                    for lk in range(2):
                        l_idx = li * 2 + lk
                        val = W_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_k = 2 * v + lk
                            W[g_k * n + g_i][col] += val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
