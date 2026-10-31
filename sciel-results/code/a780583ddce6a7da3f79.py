import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 12.0

    n, m, p = DIMENSIONS
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 7x2x2 block coefficients
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    # s=0 (M1)
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    # s=1 (M2)
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    # s=2 (M3)
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    # s=3 (M4)
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    # s=4 (M5)
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    # s=5 (M6)
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6 (M7)
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

    # Precompute nonzero entries for each factor to speed up construction
    # For U: U[i*4+j, 7*s+t] = beta[s, i//2, j//2] * U2[i%2*2 + j%2, t]
    # For V: V[j*4+k, 7*s+t] = gamma[s, j//2, k//2] * V2[j%2*2 + k%2, t]
    # For W: W[k*4+i, 7*s+t] = delta[s, i//2, k//2] * W2[k%2*2 + i%2, t]

    # Build U efficiently using vectorized operations
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            # For each block position (I, J) where beta[s,I,J] != 0
            for I in range(2):
                for J in range(2):
                    bval = beta[s, I, J]
                    if bval == 0:
                        continue
                    # For each inner position (i_prime, j_prime) where U2[i_prime*2+j_prime, t] != 0
                    for ip in range(2):
                        for jp in range(2):
                            uval = U2[ip * 2 + jp, t]
                            if uval == 0:
                                continue
                            row = (I * 2 + ip) * 4 + (J * 2 + jp)
                            U[row, col] = bval * uval

    # Build V
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for J in range(2):
                for K in range(2):
                    gval = gamma[s, J, K]
                    if gval == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            vval = V2[jp * 2 + kp, t]
                            if vval == 0:
                                continue
                            row = (J * 2 + jp) * 4 + (K * 2 + kp)
                            V[row, col] = gval * vval

    # Build W
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for I in range(2):
                for K in range(2):
                    dval = delta[s, I, K]
                    if dval == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            wval = W2[kp * 2 + ip, t]
                            if wval == 0:
                                continue
                            row = (K * 2 + kp) * 4 + (I * 2 + ip)
                            W[row, col] = dval * wval

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
