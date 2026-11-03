import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition for 2x2 blocks of size 3x3.
    # Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189.
    rank = 189
    
    # U, V, W shapes: (36, 189)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    # beta[s][I][J]: coeff of A[I, J] in L_A[s]
    # gamma[s][K][L]: coeff of B[K, L] in L_B[s]
    # alpha[s][I][K]: coeff of P[s] in C[I, K]
    # I, J, K, L are block indices in {0, 1}
    # s is step in 0..6
    
    beta = [
        [[1, 0], [0, 1]], # M1 = (A00 + A11)
        [[0, 0], [1, 1]], # M2 = (A10 + A11)
        [[1, 0], [0, 0]], # M3 = A00
        [[0, 0], [0, 1]], # M4 = A11
        [[1, 1], [0, 0]], # M5 = (A00 + A01)
        [[-1, 0], [1, 0]],# M6 = (A10 - A00)
        [[0, 1], [0, -1]] # M7 = (A01 - A11)
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # (B00 + B11)
        [[1, 0], [0, 0]], # B00
        [[0, 1], [0, -1]],# (B01 - B11)
        [[0, -1], [1, 0]],# (B10 - B00)
        [[0, 0], [0, 1]], # B11
        [[1, 1], [0, 0]], # (B00 + B01)
        [[0, 0], [1, 1]]  # (B10 + B11)
    ]
    
    alpha = [
        [[1, 0], [0, 1]], # C00+=M1, C11+=M1
        [[0, 0], [1, -1]],# C10+=M2, C11-=M2
        [[0, 1], [0, 1]], # C01+=M3, C11+=M3
        [[1, 0], [1, 0]], # C00+=M4, C10+=M4
        [[-1, 1], [0, 0]],# C00-=M5, C01+=M5
        [[0, 0], [0, 1]], # C11+=M6
        [[1, 0], [0, 0]]  # C00+=M7
    ]
    
    block_size = 3
    
    # Iterate over Strassen steps
    for s in range(7):
        # Iterate over local indices in the 3x3 blocks
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    # Column index in the rank-189 decomposition
                    col = s * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    # Update U (A components)
                    # U index: (I * 3 + i_l) * 6 + (J * 3 + j_l)
                    for I in range(2):
                        for J in range(2):
                            coeff = beta[s][I][J]
                            if coeff != 0:
                                i_glob = I * block_size + i_l
                                j_glob = J * block_size + j_l
                                U[i_glob * m + j_glob][col] += coeff
                                
                    # Update V (B components)
                    # V index: (K * 3 + j_l) * 6 + (L * 3 + k_l)
                    # Note: B is m x p. m=6, p=6.
                    # Block (K, L) is at row K, col L.
                    # Local row j_l, local col k_l.
                    for K in range(2):
                        for L in range(2):
                            coeff = gamma[s][K][L]
                            if coeff != 0:
                                j_glob = K * block_size + j_l
                                k_glob = L * block_size + k_l
                                V[j_glob * p + k_glob][col] += coeff
                                
                    # Update W (C components)
                    # W index: (L * 3 + k_l) * 6 + (I * 3 + i_l)
                    # C is p x n. p=6, n=6.
                    # Block (I, K) contributes to C[I, K].
                    # Wait, C[I, K] is row I, col K.
                    # W index is k*n + i.
                    # So W index = (K * 3 + k_l) * n + (I * 3 + i_l)
                    for I in range(2):
                        for K in range(2):
                            coeff = alpha[s][I][K]
                            if coeff != 0:
                                i_glob = I * block_size + i_l
                                k_glob = K * block_size + k_l
                                W[k_glob * n + i_glob][col] += coeff
                                
    return {"U": U, "V": V, "W": W}
