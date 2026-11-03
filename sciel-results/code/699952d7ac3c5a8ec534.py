"""Strassen-based rank-36 decomposition for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen factors for M_{2,2,2} (rank 7)
    sU = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0],
    ]
    sV = [
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    sW = [
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    
    col = 0
    # 4 Strassen blocks: m_off in {0,2}, p_off in {0,2}
    for m_off in [0, 2]:
        for p_off in [0, 2]:
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + (m_off + j)][col] = sU[i * 2 + j][s]
                for j in range(2):
                    for k in range(2):
                        V[(m_off + j) * p + (p_off + k)][col] = sV[j * 2 + k][s]
                for k in range(2):
                    for i in range(2):
                        W[(p_off + k) * n + i][col] = sW[k * 2 + i][s]
                col += 1
    
    # 2 schoolbook blocks for the 5th column (p=4): m_off in {0,2}
    # M_{2,2,1} schoolbook has rank 4: for each (i,j) pair, one term.
    for m_off in [0, 2]:
        for i in range(2):
            for j in range(2):
                U[i * m + (m_off + j)][col] = 1
                V[(m_off + j) * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
