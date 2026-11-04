import numpy as np
import time

def construct():
    """
    Rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    Uses nested Strassen (2x2x2 rank-7 applied twice for 4x4x4 rank-49)
    with direct sum over the two output halves.
    """
    t0 = time.time()
    deadline = t0 + 145.0

    # Base 2x2x2 Strassen factors
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Build 4x4x4 via two-level Kronecker structure
    # For each outer rank ro and inner rank ri, combined rank r = ro*7+ri
    # U4[i*4+j, r] = U2[I*2+J, ro] * U2[ip*2+jp, ri] where i=2I+ip, j=2J+jp
    # V4[j*4+k, r] = V2[J*2+K, ro] * V2[jp*2+kp, ri] where j=2J+jp, k=2K+kp
    # W4[k*4+i, r] = W2[K*2+I, ro] * W2[kp*2+ip, ri] where k=2K+kp, i=2I+ip

    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # U4: i = 2*I + ip, j = 2*J + jp
            # U4[i*4+j, r] = U2[2*I+J, ro] * U2[2*ip+jp, ri]
            for I in range(2):
                for J in range(2):
                    uo = U2[2*I + J, ro]
                    if uo == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            ui = U2[2*ip + jp, ri]
                            if ui == 0:
                                continue
                            i = 2*I + ip
                            j = 2*J + jp
                            U4[i*4 + j, r] = uo * ui

            # V4: j = 2*J + jp, k = 2*K + kp
            # V4[j*4+k, r] = V2[2*J+K, ro] * V2[2*jp+kp, ri]
            for J in range(2):
                for K in range(2):
                    vo = V2[2*J + K, ro]
                    if vo == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            vi = V2[2*jp + kp, ri]
                            if vi == 0:
                                continue
                            j = 2*J + jp
                            k = 2*K + kp
                            V4[j*4 + k, r] = vo * vi

            # W4: k = 2*K + kp, i = 2*I + ip
            # W4[k*4+i, r] = W2[2*K+I, ro] * W2[2*kp+ip, ri]
            for K in range(2):
                for I in range(2):
                    wo = W2[2*K + I, ro]
                    if wo == 0:
                        continue
                    for kp in range(2):
                        for ip in range(2):
                            wi = W2[2*kp + ip, ri]
                            if wi == 0:
                                continue
                            k = 2*K + kp
                            i = 2*I + ip
                            W4[k*4 + i, r] = wo * wi

    # Assemble 4x4x8: direct sum of two 4x4x4 blocks
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # Left half (output columns k=0..3)
    U[:, :49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :49] = V4[j*4 + k, :]
            W[k*4 + j, :49] = W4[k*4 + j, :]

    # Right half (output columns k=4..7)
    U[:, 49:] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + 4 + k, 49:] = V4[j*4 + k, :]
            W[(4+k)*4 + j, 49:] = W4[k*4 + j, :]

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
