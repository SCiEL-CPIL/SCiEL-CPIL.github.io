"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2 matrix multiplication
    # Each step is a tuple of (U_local, V_local, W_local)
    # Local indices: 0=A00, 1=A01, 2=A10, 3=A11 for U
    #               0=B00, 1=B01, 2=B10, 3=B11 for V
    #               0=C00, 1=C01, 2=C10, 3=C11 for W
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Block definitions
    # P1: A11 * B11 -> C11 (left block, top half)
    # A11: rows 0-1, cols 0-1. Global A indices: 0, 1, 4, 5
    # B11: rows 0-1, cols 0-1. Global B indices: 0, 1, 4, 5
    # C11: rows 0-1, cols 0-1. Global C indices: 0, 2, 1, 3
    blocks = [
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3]), # P1
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3]), # P2
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7]), # P3
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])  # P4
    ]
    
    col = 0
    for a_map, b_map, c_map in blocks:
        for u_loc, v_loc, w_loc in steps:
            for i in range(4):
                if u_loc[i] != 0:
                    U[a_map[i]][col] = u_loc[i]
                if v_loc[i] != 0:
                    V[b_map[i]][col] = v_loc[i]
                if w_loc[i] != 0:
                    W[c_map[i]][col] = w_loc[i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
