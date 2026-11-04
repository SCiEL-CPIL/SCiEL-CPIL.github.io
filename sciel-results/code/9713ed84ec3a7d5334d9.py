import numpy as np

def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses two levels of Strassen's algorithm.
    Level 1: 4x4x4 decomposition with rank 49.
    Level 2: Two independent 4x4x4 multiplications for the width 8, total rank 98.
    """
    n, m, p = 4, 4, 8
    
    # --- Level 1: Rank-49 decomposition for 4x4x4 ---
    
    # Strassen coefficients for 2x2x2 (Rank 7)
    # alpha[t][P], beta[t][Q], kappa[t][PQ]
    # t in 1..7, P,Q in 0..3 (0=00, 1=01, 2=10, 3=11)
    
    alpha = np.zeros((7, 4), dtype=int)
    beta = np.zeros((7, 4), dtype=int)
    kappa = np.zeros((7, 4), dtype=int)
    
    # t=1
    alpha[0, 0] = 1; alpha[0, 3] = 1
    beta[0, 0] = 1; beta[0, 3] = 1
    kappa[0, 0] = 1; kappa[0, 3] = 1
    
    # t=2
    alpha[1, 2] = 1; alpha[1, 3] = 1
    beta[1, 0] = 1
    kappa[1, 2] = 1; kappa[1, 3] = -1
    
    # t=3
    alpha[2, 0] = 1
    beta[2, 1] = 1; beta[2, 3] = -1
    kappa[2, 1] = 1; kappa[2, 3] = 1
    
    # t=4
    alpha[3, 3] = 1
    beta[3, 2] = 1; beta[3, 0] = -1
    kappa[3, 0] = 1; kappa[3, 2] = 1
    
    # t=5
    alpha[4, 0] = 1; alpha[4, 1] = 1
    beta[4, 3] = 1
    kappa[4, 0] = -1; kappa[4, 1] = 1
    
    # t=6
    alpha[5, 2] = 1; alpha[5, 0] = -1
    beta[5, 0] = 1; beta[5, 1] = 1
    kappa[5, 3] = 1
    
    # t=7
    alpha[6, 1] = 1; alpha[6, 3] = -1
    beta[6, 2] = 1; beta[6, 3] = 1
    kappa[6, 0] = 1

    # Inner Strassen (for 2x2x2)
    # gamma[s][ab], delta_u[s][ab], delta_v[s][ab]
    # s in 1..7, a,b in 0..1
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta_u = np.zeros((7, 2, 2), dtype=int)
    delta_v = np.zeros((7, 2, 2), dtype=int)
    
    # s=1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta_u[0, 0, 0] = 1; delta_u[0, 1, 1] = 1
    delta_v[0, 0, 0] = 1; delta_v[0, 1, 1] = 1
    
    # s=2
    gamma[1, 1, 0] = 1; gamma[1, 1, 1] = -1
    delta_u[1, 1, 0] = 1; delta_u[1, 1, 1] = 1
    delta_v[1, 0, 0] = 1
    
    # s=3
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = 1
    delta_u[2, 0, 0] = 1
    delta_v[2, 0, 1] = 1; delta_v[2, 1, 1] = -1
    
    # s=4
    gamma[3, 0, 0] = 1; gamma[3, 1, 0] = 1
    delta_u[3, 1, 1] = 1
    delta_v[3, 1, 0] = 1; delta_v[3, 0, 0] = -1
    
    # s=5
    gamma[4, 0, 1] = 1; gamma[4, 0, 0] = -1
    delta_u[4, 0, 0] = 1; delta_u[4, 0, 1] = 1
    delta_v[4, 1, 1] = 1
    
    # s=6
    gamma[5, 1, 1] = 1
    delta_u[5, 1, 0] = 1; delta_u[5, 0, 0] = -1
    delta_v[5, 0, 0] = 1; delta_v[5, 0, 1] = 1
    
    # s=7
    gamma[6, 0, 0] = 1
    delta_u[6, 0, 1] = 1; delta_u[6, 1, 1] = -1
    delta_v[6, 1, 0] = 1; delta_v[6, 1, 1] = 1

    # Build 4x4x4 factors (16x49)
    U444 = np.zeros((16, 49), dtype=float)
    V444 = np.zeros((16, 49), dtype=float)
    W444 = np.zeros((16, 49), dtype=float)
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # U444: A indices
            for a in range(2):
                for b in range(2):
                    coeff_u = delta_u[s, a, b]
                    if coeff_u == 0: continue
                    for P in range(4):
                        coeff_a = alpha[t, P]
                        if coeff_a == 0: continue
                        coeff_total = coeff_u * coeff_a
                        if coeff_total == 0: continue
                        P_row = P // 2
                        P_col = P % 2
                        i = 2 * P_row + a
                        j = 2 * P_col + b
                        U444[i * 4 + j, r] += coeff_total
            
            # V444: B indices
            for a in range(2):
                for b in range(2):
                    coeff_v = delta_v[s, a, b]
                    if coeff_v == 0: continue
                    for Q in range(4):
                        coeff_b = beta[t, Q]
                        if coeff_b == 0: continue
                        coeff_total = coeff_v * coeff_b
                        if coeff_total == 0: continue
                        Q_row = Q // 2
                        Q_col = Q % 2
                        j = 2 * Q_row + a
                        k = 2 * Q_col + b
                        V444[j * 4 + k, r] += coeff_total
            
            # W444: C indices
            for a in range(2):
                for b in range(2):
                    coeff_m = gamma[s, a, b]
                    if coeff_m == 0: continue
                    for PQ in range(4):
                        coeff_k = kappa[t, PQ]
                        if coeff_k == 0: continue
                        coeff_total = coeff_m * coeff_k
                        if coeff_total == 0: continue
                        PQ_row = PQ // 2
                        PQ_col = PQ % 2
                        i = 2 * PQ_row + a
                        k = 2 * PQ_col + b
                        W444[k * 4 + i, r] += coeff_total

    # --- Level 2: 4x4x8 factors ---
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Part 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            
    # Part 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8*j + k, 49:98] = V444[4*j + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4*k + i, 49:98] = W444[4*(k-4) + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
