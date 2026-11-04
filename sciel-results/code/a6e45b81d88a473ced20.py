import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    # Precompute nonzero indices for efficiency
    U2_nz = [(i, j, U2[i, j]) for i in range(4) for j in range(7) if U2[i, j] != 0]
    V2_nz = [(i, j, V2[i, j]) for i in range(4) for j in range(7) if V2[i, j] != 0]
    W2_nz = [(i, j, W2[i, j]) for i in range(4) for j in range(7) if W2[i, j] != 0]

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri

            # U factor: U4[i*4+j, r]
            for uo_idx, uo_j, uo_val in U2_nz:
                if uo_j != ro:
                    continue
                I = uo_idx // 2
                J = uo_idx % 2
                for ui_idx, ui_j, ui_val in U2_nz:
                    if ui_j != ri:
                        continue
                    ip = ui_idx // 2
                    jp = ui_idx % 2
                    i = 2 * I + ip
                    j = 2 * J + jp
                    U4[i * 4 + j, r] += uo_val * ui_val

            # V factor: V4[j*4+k, r]
            for vo_idx, vo_j, vo_val in V2_nz:
                if vo_j != ro:
                    continue
                J = vo_idx // 2
                K = vo_idx % 2
                for vi_idx, vi_j, vi_val in V2_nz:
                    if vi_j != ri:
                        continue
                    jp = vi_idx // 2
                    kp = vi_idx % 2
                    j = 2 * J + jp
                    k = 2 * K + kp
                    V4[j * 4 + k, r] += vo_val * vi_val

            # W factor: W4[k*4+i, r]
            for wo_idx, wo_j, wo_val in W2_nz:
                if wo_j != ro:
                    continue
                K = wo_idx // 2
                I = wo_idx % 2
                for wi_idx, wi_j, wi_val in W2_nz:
                    if wi_j != ri:
                        continue
                    kp = wi_idx // 2
                    ip = wi_idx % 2
                    k = 2 * K + kp
                    i = 2 * I + ip
                    W4[k * 4 + i, r] += wo_val * wi_val

        if time.time() > deadline:
            # Fallback to naive if Strassen construction times out
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U[i * m + j, r] = 1
                        V[j * p + k, r] = 1
                        W[k * n + i, r] = 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct 4x4x8 decomposition (Rank 98)
    # The 4x4x8 tensor is the direct sum of two 4x4x4 tensors
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    # Left block (output columns 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]

    # Right block (output columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + 4 + k, 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
