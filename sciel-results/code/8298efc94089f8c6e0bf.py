import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Scalar Strassen coefficients
    # s is 0..6 (1..7)
    # p, q are 0..1
    # sigma_A[s][p][q]
    sigma_A = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [-1, 0]] # 7
    ]
    
    sigma_B = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]],# 3
        [[0, -1], [1, 0]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 1], [1, 1]]  # 7
    ]
    
    # gamma[s][p][q] is coeff of M_s in C_pq
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    gamma = [
        [[1, 0], [0, 0]], # s=1: C00=1, C01=0, C10=0, C11=1 -> Wait, gamma[s] is 2x2 matrix of coeffs for C_pq
        # Let's define gamma[s][p][q] as coeff of M_s in C_{p,q}
        # p is row of C (I), q is col of C (K)
        # So gamma[s][I][K]
        [[1, 0], [0, 1]], # s=1: C00+=1, C11+=1
        [[0, 0], [1, -1]],# s=2: C10+=1, C11-=1
        [[0, 1], [0, 1]], # s=3: C01+=1, C11+=1
        [[1, 0], [1, 0]], # s=4: C00+=1, C10+=1
        [[-1, 1], [0, 0]],# s=5: C00-=1, C01+=1
        [[0, 0], [0, 1]], # s=6: C11+=1
        [[1, 0], [0, 0]]  # s=7: C00+=1
    ]
    
    # Block Strassen coefficients are the same as scalar ones
    # alpha[T][I][J] = sigma_A[T][I][J]
    # beta[T][J][K] = sigma_B[T][J][K]
    # delta[T][I][K] = gamma[T][I][K]
    
    for T in range(7):
        for s in range(7):
            r = T * 7 + s
            
            # Update U
            # U[i, j, r] where i=2I+p, j=2J+q
            for I in range(2):
                for J in range(2):
                    alpha_IJ = sigma_A[T][I][J]
                    if alpha_IJ != 0:
                        for p in range(2):
                            for q in range(2):
                                sigma_pq = sigma_A[s][p][q]
                                if sigma_pq != 0:
                                    i = 2 * I + p
                                    j = 2 * J + q
                                    U[i * m + j, r] += alpha_IJ * sigma_pq
                                    
            # Update V
            # V[j, k, r] where j=2J+p, k=2K+q
            for J in range(2):
                for K in range(2):
                    beta_JK = sigma_B[T][J][K]
                    if beta_JK != 0:
                        for p in range(2):
                            for q in range(2):
                                sigma_pq = sigma_B[s][p][q]
                                if sigma_pq != 0:
                                    j = 2 * J + p
                                    k = 2 * K + q
                                    V[j * p + k, r] += beta_JK * sigma_pq
                                    
            # Update W
            # W[k, i, r] where k=2K+q, i=2I+p
            for I in range(2):
                for K in range(2):
                    delta_IK = gamma[T][I][K]
                    if delta_IK != 0:
                        for p in range(2):
                            for q in range(2):
                                gamma_pq = gamma[s][p][q]
                                if gamma_pq != 0:
                                    i = 2 * I + p
                                    k = 2 * K + q
                                    W[k * n + i, r] += delta_IK * gamma_pq

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
