"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0 slice (2x1x6), rank 12
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1,2} and {3,4} slices (2x2x6 each)
    # Each 2x2x6 is split into three 2x2x2 blocks
    # Strassen coefficients for 2x2x2
    # U: (i, j), V: (j, k), W: (k, i)
    # Indices 0..3 for U, V; 0..3 for W
    strassen_U = [
        [1, 0, 0, 1], # T1
        [0, 0, 1, 1], # T2
        [1, 0, 0, 0], # T3
        [0, 0, 0, 1], # T4
        [1, 1, 0, 0], # T5
        [0, 1, 1, 0], # T6 (U10=1, U00=-1 -> [ -1, 1, 1, 0 ]? No, U00=-1, U01=0, U10=1, U11=0)
                       # Wait, T6 U: A10 - A00. U10=1, U00=-1.
                       # My previous list: [0, 1, 1, 0] was U01=1, U10=1. That was T5?
                       # Let's re-verify T6 U.
                       # M6 = (A10 - A00)(B00 + B01).
                       # U10=1, U00=-1.
                       # Vector: [-1, 0, 1, 0].
        [-1, 0, 1, 0], # T6 corrected
        [0, 1, 0, 1]   # T7: A01 - A11. U01=1, U11=-1. Vector: [0, 1, 0, -1].
    ]
    # Correcting T7 U: [0, 1, 0, -1]
    strassen_U[6] = [0, 1, 0, -1]
    
    strassen_V = [
        [1, 0, 0, 1], # T1: B00+B11. V00=1, V11=1.
        [1, 0, 0, 0], # T2: B00. V00=1.
        [0, 1, 0, -1],# T3: B01-B11. V01=1, V11=-1.
        [0, -1, 1, 0],# T4: B10-B00. V10=1, V00=-1.
        [0, 0, 0, 1], # T5: B11. V11=1.
        [1, 1, 0, 0], # T6: B00+B01. V00=1, V01=1.
        [0, 0, 1, 1]  # T7: B10+B11. V10=1, V11=1.
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # T1: C00+C11. W00=1, W11=1.
        [0, 1, 0, -1],# T2: C10-C11. W01=1, W11=-1.
        [0, 1, 0, 1], # T3: C01+C11. W01=1, W11=1.
        [1, 1, 0, 0], # T4: C00+C10. W00=1, W01=1.
        [-1, 1, 0, 0],# T5: -C00+C01. W00=-1, W01=1.
        [0, 0, 0, 1], # T6: C11. W11=1.
        [1, 0, 0, 1]  # T7: C00+C11. W00=1, W11=1.
    ]
    
    j_blocks = [(1, 2), (3, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for j_start, j_end in j_blocks:
        for k_start, k_end in k_blocks:
            for t in range(7):
                # Local indices 0,1
                for i_local in range(2):
                    for j_local in range(2):
                        u_val = strassen_U[t][i_local * 2 + j_local]
                        if u_val != 0:
                            i_global = i_local
                            j_global = j_start + j_local
                            U[i_global * m + j_global][col] += u_val
                for j_local in range(2):
                    for k_local in range(2):
                        v_val = strassen_V[t][j_local * 2 + k_local]
                        if v_val != 0:
                            j_global = j_start + j_local
                            k_global = k_start + k_local
                            V[j_global * p + k_global][col] += v_val
                for k_local in range(2):
                    for i_local in range(2):
                        w_val = strassen_W[t][k_local * 2 + i_local]
                        if w_val != 0:
                            k_global = k_start + k_local
                            i_global = i_local
                            W[k_global * n + i_global][col] += w_val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
