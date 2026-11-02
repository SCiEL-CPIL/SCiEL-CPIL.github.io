"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add contributions to a column
    def add_col(col, u_idx_vals, v_idx_vals, w_idx_vals):
        for idx, val in u_idx_vals:
            U[idx][col] = val
        for idx, val in v_idx_vals:
            V[idx][col] = val
        for idx, val in w_idx_vals:
            W[idx][col] = val

    # Block 0 (k=0,1)
    # M1
    add_col(0, [(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)])
    # M2
    add_col(1, [(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)])
    # M3
    add_col(2, [(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)])
    # M4
    add_col(3, [(3, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)])
    # M5
    add_col(4, [(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)])
    # M6
    add_col(5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    # M7
    add_col(6, [(1, 1), (3, -1)], [(4, 1), (5, 1)], [(0, 1)])

    # Block 1 (k=2,3)
    # M1
    add_col(7, [(0, 1), (3, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)])
    # M2
    add_col(8, [(2, 1), (3, 1)], [(2, 1)], [(5, 1), (7, -1)])
    # M3
    add_col(9, [(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)])
    # M4
    add_col(10, [(3, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)])
    # M5
    add_col(11, [(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)])
    # M6
    add_col(12, [(2, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)])
    # M7
    add_col(13, [(1, 1), (3, -1)], [(6, 1), (7, 1)], [(4, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
