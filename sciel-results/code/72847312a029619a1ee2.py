"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block-wise Strassen decomposition for 6x6 matrix multiplication.
    # Splits into 2x2 blocks of 3x3 matrices.
    # Rank = 7 * (3*3*3) = 189.
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[k][u][v] is coefficient of A[u][v] in A'_k
    # beta[k][u][v] is coefficient of B[u][v] in B'_k
    # gamma[k][u][v] is coefficient of M_k in C[u][v]
    
    # k=0: M1 = (A00+A11)(B00+B11)
    alpha = [
        [[1, 0], [1, 1]], # k=0 (M1)
        [[0, 0], [1, 1]], # k=1 (M2)
        [[1, 0], [0, 0]], # k=2 (M3)
        [[0, 0], [0, 1]], # k=3 (M4)
        [[1, 1], [0, 0]], # k=4 (M5)
        [[-1, 0], [1, 0]],# k=5 (M6)
        [[0, 1], [0, -1]] # k=6 (M7)
    ]
    beta = [
        [[1, 0], [0, 1]], # k=0 (M1)
        [[1, 0], [0, 0]], # k=1 (M2)
        [[0, 1], [0, -1]],# k=2 (M3)
        [[-1, 0], [1, 0]],# k=3 (M4)
        [[0, 0], [0, 1]], # k=4 (M5)
        [[1, 1], [0, 0]], # k=5 (M6)
        [[0, 0], [1, 1]]  # k=6 (M7)
    ]
    gamma = [
        [[1, 0], [0, 0]], # k=0 (M1) -> C00, C11
        [[0, 0], [0, -1]],# k=1 (M2) -> C11
        [[0, 1], [1, 0]], # k=2 (M3) -> C01, C10
        [[-1, 1], [0, 0]],# k=3 (M4) -> C00, C01
        [[1, 0], [0, 0]], # k=4 (M5) -> C00
        [[0, 0], [0, 1]], # k=5 (M6) -> C11
        [[1, 0], [0, 1]]  # k=6 (M7) -> C00, C11
    ]
    # Correction for gamma based on formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 + M7
    
    gamma = [
        [[1, 0], [0, 1]], # M1: C00=1, C11=1
        [[0, 0], [0, -1]],# M2: C11=-1
        [[0, 1], [1, 0]], # M3: C01=1, C10=1
        [[-1, 1], [0, 0]],# M4: C00=-1, C01=1
        [[1, 0], [0, 0]], # M5: C00=1
        [[0, 0], [0, 1]], # M6: C11=1
        [[1, 0], [0, 1]]  # M7: C00=1, C11=1
    ]
    
    # Iterate over Strassen steps and schoolbook indices
    for k in range(7):
        for a in range(3):
            for c in range(3):
                for b in range(3):
                    r = k * 27 + a * 9 + c * 3 + b
                    
                    # Fill U
                    for i in range(6):
                        if i % 3 != a: continue
                        for j in range(6):
                            if j % 3 != c: continue
                            u = i // 3
                            v = j // 3
                            U[i * 6 + j][r] = alpha[k][u][v]
                    
                    # Fill V
                    for i in range(6):
                        if i % 3 != c: continue
                        for j in range(6):
                            if j % 3 != b: continue
                            u = i // 3
                            v = j // 3
                            V[i * 6 + j][r] = beta[k][u][v]
                            
                    # Fill W
                    # W index is k_global * n + i_global
                    # Contribution to C[i_global, k_global]
                    for i in range(6):
                        if i % 3 != a: continue
                        for k_g in range(6):
                            if k_g % 3 != b: continue
                            u = i // 3
                            v = k_g // 3
                            W[k_g * 6 + i][r] = gamma[k][u][v]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
