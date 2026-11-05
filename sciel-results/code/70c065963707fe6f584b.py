"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    
    # Partitions
    I1 = [0, 1]
    I2 = [2]
    J1 = [0, 1]
    J2 = [2, 3, 4]
    K1 = [0, 1]
    K2 = [2, 3, 4]
    
    blocks = [
        (I1, J1, K1), # 2x2x2
        (I1, J2, K1), # 2x3x2
        (I2, J1, K1), # 1x2x2
        (I1, J1, K2), # 2x2x3
        (I1, J2, K2), # 2x3x3
        (I2, J1, K2), # 1x2x3
        (I2, J2, K1), # 1x3x2
        (I2, J2, K2)  # 1x3x3
    ]
    
    # Calculate total rank
    total_rank = 0
    for I, J, K in blocks:
        n_L = len(I)
        m_L = len(J)
        p_L = len(K)
        if n_L == 2 and m_L == 2 and p_L == 2:
            total_rank += 7
        else:
            total_rank += n_L * m_L * p_L
            
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    for I, J, K in blocks:
        n_L = len(I)
        m_L = len(J)
        p_L = len(K)
        
        if n_L == 2 and m_L == 2 and p_L == 2:
            # Strassen Decomposition
            # Local indices il, jl, kl in {0, 1}
            # Map to global i, j, k
            # I = [0, 1], J = [0, 1], K = [0, 1]
            
            # Coefficients for Strassen
            # alpha[r][il][jl]
            # beta[r][jl][kl]
            # gamma[r][kl][il]
            
            alphas = [
                [[1, 0], [0, 1]], # M1: A00 + A11
                [[0, 0], [1, 1]], # M2: A10 + A11
                [[1, 0], [0, 0]], # M3: A00
                [[0, 0], [0, 1]], # M4: A11
                [[1, 1], [0, 0]], # M5: A00 + A01
                [[-1, 0], [1, 0]],# M6: -A00 + A10
                [[0, 1], [0, -1]] # M7: A01 - A11
            ]
            betas = [
                [[1, 0], [0, 1]], # M1: B00 + B11
                [[1, 0], [0, 0]], # M2: B00
                [[0, 1], [0, -1]],# M3: B01 - B11
                [[-1, 0], [1, 0]],# M4: -B00 + B10
                [[0, 0], [0, 1]], # M5: B11
                [[1, 1], [0, 0]], # M6: B00 + B01
                [[0, 0], [1, 1]]  # M7: B10 + B11
            ]
            gammas = [
                [[1, 0], [0, 1]], # M1: C00 + C11
                [[0, 1], [-1, 0]],# M2: C10 - C11
                [[0, 0], [1, 1]], # M3: C01 + C11
                [[1, 0], [1, 0]], # M4: C00 + C10
                [[-1, 0], [1, 0]],# M5: -C00 + C01
                [[0, 0], [0, 1]], # M6: C11
                [[1, 0], [0, 1]]  # M7: C00 + C11
            ]
            
            for r in range(7):
                rank_idx = current_rank + r
                for il in range(2):
                    for jl in range(2):
                        if alphas[r][il][jl] != 0:
                            i = I[il]
                            j = J[jl]
                            U[i * m + j][rank_idx] = alphas[r][il][jl]
                for jl in range(2):
                    for kl in range(2):
                        if betas[r][jl][kl] != 0:
                            j = J[jl]
                            k = K[kl]
                            V[j * p + k][rank_idx] = betas[r][jl][kl]
                for kl in range(2):
                    for il in range(2):
                        if gammas[r][kl][il] != 0:
                            k = K[kl]
                            i = I[il]
                            W[k * n + i][rank_idx] = gammas[r][kl][il]
            
            current_rank += 7
        else:
            # Schoolbook Decomposition
            for il in range(n_L):
                for jl in range(m_L):
                    for kl in range(p_L):
                        rank_idx = current_rank
                        i = I[il]
                        j = J[jl]
                        k = K[kl]
                        U[i * m + j][rank_idx] = 1
                        V[j * p + k][rank_idx] = 1
                        W[k * n + i][rank_idx] = 1
                        current_rank += 1
                        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
