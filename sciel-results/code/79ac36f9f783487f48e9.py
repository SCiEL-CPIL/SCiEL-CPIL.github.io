"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        for idx in u_indices:
            U[idx[0]][col] += idx[1]
        for idx in v_indices:
            V[idx[0]][col] += idx[1]
        for idx in w_indices:
            W[idx[0]][col] += idx[1]

    # Strassen terms for 2x2 * 2x2
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices (Block 1): 00=0, 01=1, 10=4, 11=5
    # C indices (Block 1): 00=0, 01=2, 10=1, 11=3
    
    # Term 1: M1
    add_term([(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # Term 2: M2
    add_term([(2, 1), (3, 1)], [(0, 1)], [(1, 1)], 1)
    # Term 3: M3
    add_term([(0, 1)], [(4, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    # Term 4: M4
    add_term([(3, 1)], [(1, 1), (5, 1)], [(0, 1), (1, 1)], 3)
    # Term 5: M5
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    # Term 6: M6
    add_term([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    # Term 7: M7
    add_term([(1, 1), (3, -1)], [(4, 1), (5, 1)], [(0, -1)], 6)

    # B indices (Block 2): 02=2, 03=3, 12=6, 13=7
    # C indices (Block 2): 02=4, 03=6, 12=5, 13=7

    # Term 8: M1'
    add_term([(0, 1), (3, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    # Term 9: M2'
    add_term([(2, 1), (3, 1)], [(2, 1)], [(5, 1)], 8)
    # Term 10: M3'
    add_term([(0, 1)], [(6, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    # Term 11: M4'
    add_term([(3, 1)], [(3, 1), (7, 1)], [(4, 1), (5, 1)], 10)
    # Term 12: M5'
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    # Term 13: M6'
    add_term([(2, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    # Term 14: M7'
    add_term([(1, 1), (3, -1)], [(6, 1), (7, 1)], [(4, -1)], 13)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
