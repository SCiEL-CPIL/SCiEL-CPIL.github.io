"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank estimate
    # M_344: 9 + 3 + 3 + 3 = 18
    # M_343: 9 + 3 = 12
    # Total 30
    R = 30
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    current_rank = 0
    
    def add_schoolblock(i_off, j_off, k_off, n_sub, m_sub, p_sub):
        nonlocal current_rank
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    r = current_rank
                    current_rank += 1
                    # U index: (i_off + i) * m + (j_off + j)
                    U[(i_off + i) * m + (j_off + j)][r] = 1
                    # V index: (j_off + j) * p + (k_off + k)
                    V[(j_off + j) * p + (k_off + k)][r] = 1
                    # W index: (k_off + k) * n + (i_off + i)
                    W[(k_off + k) * n + (i_off + i)][r] = 1

    # Part 1: 3x4 x 4x4 (k in 0..3)
    # A_L (3x3) * B_LL (3x3) -> i 0..2, j 0..2, k 0..2
    add_schoolblock(0, 0, 0, 3, 3, 3)
    
    # A_L (3x3) * B_LR (3x1) -> i 0..2, j 0..2, k 3
    add_schoolblock(0, 0, 3, 3, 3, 1)
    
    # A_R (3x1) * B_RL (1x3) -> i 0..2, j 3, k 0..2
    add_schoolblock(0, 3, 0, 3, 1, 3)
    
    # A_R (3x1) * B_RR (1x1) -> i 0..2, j 3, k 3
    add_schoolblock(0, 3, 3, 3, 1, 1)
    
    # Part 2: 3x4 x 4x3 (k in 4..6)
    # A_L (3x3) * B_LL (3x3) -> i 0..2, j 0..2, k 4..6
    add_schoolblock(0, 0, 4, 3, 3, 3)
    
    # A_R (3x1) * B_RL (1x3) -> i 0..2, j 3, k 4..6
    add_schoolblock(0, 3, 4, 3, 1, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
