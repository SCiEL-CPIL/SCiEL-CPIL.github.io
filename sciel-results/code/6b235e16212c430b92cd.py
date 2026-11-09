"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 36: 4 Strassen blocks (7 each) + 2 schoolbook blocks (4 each)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Local indices: ia in {0,1}, jb in {0,1}, ka in {0,1}
    # U local: ia*2+jb, V local: jb*2+ka, W local: ka*2+ia
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, -1, 0, 1], # M3: B11-B01
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+C11
        [0, 1, 0, 0],  # M2: C10
        [0, 0, 1, 1],  # M3: C01+C11
        [1, 1, 0, 0],  # M4: C00+C10
        [-1, 0, 1, 0], # M5: -C00+C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0],  # M7: C00
    ]
    
    for jb_group in range(2):
        for ka_group in range(2):
            for r in range(7):
                for ia in range(2):
                    for jb in range(2):
                        u_idx = ia * m + (jb_group * 2 + jb)
                        U[u_idx][col] += strassen_U[r][ia * 2 + jb]
                for jb in range(2):
                    for ka in range(2):
                        v_idx = (jb_group * 2 + jb) * p + (ka_group * 2 + ka)
                        V[v_idx][col] += strassen_V[r][jb * 2 + ka]
                for ka in range(2):
                    for ia in range(2):
                        w_idx = (ka_group * 2 + ka) * n + ia
                        W[w_idx][col] += strassen_W[r][ka * 2 + ia]
                col += 1
    
    # Schoolbook for k=4, j in {0,1}
    for ia in range(2):
        for jb in range(2):
            u_idx = ia * m + jb
            v_idx = jb * p + 4
            w_idx = 4 * n + ia
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Schoolbook for k=4, j in {2,3}
    for ia in range(2):
        for jb in range(2):
            u_idx = ia * m + (2 + jb)
            v_idx = (2 + jb) * p + 4
            w_idx = 4 * n + ia
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
