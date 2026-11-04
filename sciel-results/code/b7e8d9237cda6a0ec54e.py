"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block partitions: n=[1,2], m=[1,2,2], p=[1,2,2]
    n_blocks = [1, 2]
    m_blocks = [1, 2, 2]
    p_blocks = [1, 2, 2]
    
    # Compute total rank (all schoolbook for now)
    rank = 0
    for ni in n_blocks:
        for mj in m_blocks:
            for pk in p_blocks:
                rank += ni * mj * pk
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    i0 = 0
    for ni in n_blocks:
        j0 = 0
        for mj in m_blocks:
            k0 = 0
            for pk in p_blocks:
                for a in range(ni):
                    for b in range(mj):
                        for c in range(pk):
                            U[(i0+a) * m + (j0+b)][col] = 1
                            V[(j0+b) * p + (k0+c)][col] = 1
                            W[(k0+c) * n + (i0+a)][col] = 1
                            col += 1
                k0 += pk
            j0 += mj
        i0 += ni
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
