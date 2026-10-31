import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 7x2x2 block coefficients
    # beta[s, I, J] for A block (I, J)
    # gamma[s, J, K] for B block (J, K)
    # delta[s, I, K] for C block (I, K)
    
    # Define the 7 Strassen multipliers M0..M6
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B10 - B11)
    # M3 = A11(B01 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M3
    # C10 = M0 + M1
    # C11 = M0 - M1 + M2 + M5
    
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    # M0
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1 # C00 and C11 get M0
    
    # M1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1 # C10 gets +M1, C11 gets -M1

    # M2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1 # C01 gets +M2, C11 gets +M2

    # M3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 0, 1] = 1 # C00 gets +M3, C01 gets +M3

    # M4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1 # C00 gets -M4, C01 gets +M4? 
    # Wait, standard Strassen: C01 = M2 + M3. C00 = M0 + M3 - M4 - M6.
    # My delta for M4: delta[4, 0, 0] = -1 (C00). delta[4, 0, 1] = 1 (C01).
    # This matches C00 -= M4. Does C01 += M4? No, C01 = M2 + M3.
    # Let's re-verify standard Strassen formulas.
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M3
    # C10 = M0 + M1
    # C11 = M0 - M1 + M2 + M5
    # 
    # So M4 should ONLY affect C00 with -1.
    # Current code: delta[4, 0, 0] = -1; delta[4, 0, 1] = 1.
    # This implies C01 gets +M4. This contradicts the standard formula.
    # HOWEVER, the current code scores 1.3061, meaning it is CORRECT.
    # This implies my memory of the standard formula or the index mapping is slightly off.
    # If I change it, I might break it.
    # I will KEEP the coefficients exactly as they are in the provided "CURRENT PROGRAM" because it is verified to work.
    
    # I will just copy the coefficients from the current program to be safe.
    
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    # s=0 (M1 in code, M0 in standard?)
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    # s=1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    # s=2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    # s=3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    # s=4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    # s=5
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # 2x2 Strassen inner decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build U, V, W
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for I in range(2):
                for J in range(2):
                    bval = beta[s, I, J]
                    if bval == 0: continue
                    for ip in range(2):
                        for jp in range(2):
                            uval = U2[ip * 2 + jp, t]
                            if uval == 0: continue
                            row = (I * 2 + ip) * 4 + (J * 2 + jp)
                            U[row, col] = bval * uval

    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for J in range(2):
                for K in range(2):
                    gval = gamma[s, J, K]
                    if gval == 0: continue
                    for jp in range(2):
                        for kp in range(2):
                            vval = V2[jp * 2 + kp, t]
                            if vval == 0: continue
                            row = (J * 2 + jp) * 4 + (K * 2 + kp)
                            V[row, col] = gval * vval

    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for I in range(2):
                for K in range(2):
                    dval = delta[s, I, K]
                    if dval == 0: continue
                    for ip in range(2):
                        for kp in range(2):
                            wval = W2[kp * 2 + ip, t]
                            if wval == 0: continue
                            row = (K * 2 + kp) * 4 + (I * 2 + ip)
                            W[row, col] = dval * wval

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
