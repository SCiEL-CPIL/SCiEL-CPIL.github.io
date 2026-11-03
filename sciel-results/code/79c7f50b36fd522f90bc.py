"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # 4 blocks of 2x2x2 Strassen (rank 7 each) + 1 term for j=2 (rank 1)
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[i,j], v[j,k], w[i,k]
    # i,j,k in {0,1}
    strassen_terms = []
    
    # M1
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [0, 0]]
    w2 = [[0, 0], [1, -1]]
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 1], [-1, 0]]
    w3 = [[-1, 1], [0, 0]]
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = [[0, 0], [0, 1]]
    v4 = [[-1, 0], [1, 0]]
    w4 = [[1, 0], [1, 0]]
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[-1, 1], [0, 0]]
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]]
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = [[0, 1], [-1, 0]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 1]]
    strassen_terms.append((u7, v7, w7))
    
    current_col = 0
    
    # 4 blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    # j in {0,1}
    for block_idx in range(4):
        k_offset = block_idx * 2
        for u, v, w in strassen_terms:
            # u is 2x2, v is 2x2, w is 2x2
            # Map local i,j,k to global
            for i in range(2):
                for j in range(2):
                    if u[i][j] != 0:
                        U[i * m + j][current_col] = u[i][j]
            for j in range(2):
                for k in range(2):
                    if v[j][k] != 0:
                        V[j * p + (k_offset + k)][current_col] = v[j][k]
            for i in range(2):
                for k in range(2):
                    if w[i][k] != 0:
                        W[(k_offset + k) * n + i][current_col] = w[i][k]
            current_col += 1
            
    # Last term for j=2
    # U[i, 2] = 1
    # V[2, k] = 1
    # W[k, i] = 1
    for i in range(n):
        U[i * m + 2][current_col] = 1
    for k in range(p):
        V[2 * p + k][current_col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][current_col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
