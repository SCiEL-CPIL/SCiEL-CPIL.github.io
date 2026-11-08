# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U_local[s] maps (i, j) to coeff
    # V_local[s] maps (j, k) to coeff
    # W_local[s] maps (i, k) to coeff
    
    U_local = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): 1},  # P2
        {(0, 0): 1},             # P3
        {(1, 1): 1},             # P4
        {(0, 0): 1, (0, 1): 1},  # P5
        {(1, 0): 1, (0, 0): -1}, # P6
        {(0, 1): 1, (1, 1): -1}  # P7
    ]
    
    V_local = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(0, 0): 1},             # P2
        {(0, 1): 1, (1, 1): -1}, # P3
        {(1, 0): 1, (0, 0): -1}, # P4
        {(1, 1): 1},             # P5
        {(0, 0): 1, (0, 1): 1},  # P6
        {(1, 0): 1, (1, 1): 1}   # P7
    ]
    
    W_local = [
        {(0, 0): 1, (1, 1): 1},  # P1 -> C00, C11
        {(1, 0): 1, (1, 1): -1}, # P2 -> C10, C11
        {(0, 1): 1, (1, 1): 1},  # P3 -> C01, C11
        {(0, 0): 1, (1, 0): 1},  # P4 -> C00, C10
        {(0, 0): -1, (0, 1): 1}, # P5 -> C00, C01
        {(1, 1): 1},             # P6 -> C11
        {(0, 0): 1}              # P7 -> C00
    ]

    col = 0
    # A is 2x4. Split into two 2x2 blocks along columns.
    # Block a=0: cols 0,1. Block a=1: cols 2,3.
    # B is 4x6. Rows 0-1 correspond to A block 0. Rows 2-3 correspond to A block 1.
    # B is split into three 2x2 blocks along columns.
    # Block b=0: cols 0,1. Block b=1: cols 2,3. Block b=2: cols 4,5.
    
    for a in range(2):
        for b in range(3):
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_local[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_local[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_local[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
