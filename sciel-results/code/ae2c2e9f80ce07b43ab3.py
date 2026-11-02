DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # alpha[r][i][j]
    alpha = [
        [[1, 0], [0, 1]], # M1: (a+d)
        [[0, 0], [1, 1]], # M2: (c+d)
        [[1, 0], [0, 0]], # M3: a
        [[0, 0], [0, 1]], # M4: d
        [[1, 1], [0, 0]], # M5: (a+b)
        [[-1, 0], [1, 0]], # M6: (c-a)
        [[0, 1], [0, -1]]  # M7: (b-d)
    ]
    
    # beta[r][j][k]
    beta = [
        [[1, 1], [1, 1]], # M1: (e+h)
        [[1, 0], [0, 0]], # M2: e
        [[0, 1], [0, -1]], # M3: (f-h)
        [[-1, 0], [1, 0]], # M4: (g-e)
        [[0, 0], [0, 1]], # M5: h
        [[1, 1], [0, 0]], # M6: (e+f)
        [[0, 0], [1, 1]]  # M7: (g+h)
    ]
    
    # gamma[r][k][i]
    gamma = [
        [[1, 0], [0, 1]], # M1: p=1, s=1
        [[0, 0], [1, -1]], # M2: r=1, s=-1
        [[0, 0], [1, 1]], # M3: q=1, s=1
        [[1, 1], [0, 0]], # M4: p=1, r=1
        [[-1, 0], [1, 0]], # M5: p=-1, q=1
        [[0, 0], [0, 1]], # M6: s=1
        [[1, 0], [0, 0]]  # M7: p=1
    ]
    
    # Fill 6 blocks of 2x2 using Strassen
    for block_idx in range(6):
        k_start = block_idx * 2
        for r in range(7):
            rank_idx = block_idx * 7 + r
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][rank_idx] = alpha[r][i][j]
            for j in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * 13 + k_global][rank_idx] = beta[r][j][k_local]
            for k_local in range(2):
                for i in range(2):
                    k_global = k_start + k_local
                    W[k_global * 2 + i][rank_idx] = gamma[r][k_local][i]
                    
    # Fill last block of 2x1 using schoolbook
    k_last = 12
    # Term 1: A[0,0] B[0,12] -> C[0,12]
    U[0 * 2 + 0][42] = 1
    V[0 * 13 + k_last][42] = 1
    W[k_last * 2 + 0][42] = 1
    
    # Term 2: A[0,1] B[1,12] -> C[0,12]
    U[0 * 2 + 1][43] = 1
    V[1 * 13 + k_last][43] = 1
    W[k_last * 2 + 0][43] = 1
    
    # Term 3: A[1,0] B[0,12] -> C[1,12]
    U[1 * 2 + 0][44] = 1
    V[0 * 13 + k_last][44] = 1
    W[k_last * 2 + 1][44] = 1
    
    # Term 4: A[1,1] B[1,12] -> C[1,12]
    U[1 * 2 + 1][45] = 1
    V[1 * 13 + k_last][45] = 1
    W[k_last * 2 + 1][45] = 1
    
    return {"U": U, "V": V, "W": W}
