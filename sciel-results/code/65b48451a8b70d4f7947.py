import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 decomposition
    # Local indices for 2x2 matrices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    #
    # U coefficients (A side):
    # M1: (a00 + a11) -> {0: 1, 3: 1}
    # M2: (a10 + a11) -> {2: 1, 3: 1}
    # M3: a00          -> {0: 1}
    # M4: a11          -> {3: 1}
    # M5: (a00 + a01)  -> {0: 1, 1: 1}
    # M6: (a10 - a00)  -> {2: 1, 0: -1}
    # M7: (a01 - a11)  -> {1: 1, 3: -1}
    #
    # V coefficients (B side):
    # M1: (b00 + b11) -> {0: 1, 3: 1}
    # M2: b00          -> {0: 1}
    # M3: (b01 - b11)  -> {1: 1, 3: -1}
    # M4: (b10 - b00)  -> {2: 1, 0: -1}
    # M5: b11          -> {3: 1}
    # M6: (b00 + b01)  -> {0: 1, 1: 1}
    # M7: (b10 + b11)  -> {2: 1, 3: 1}
    #
    # W coefficients (C side):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1: {C00: 1, C11: 1} -> {0: 1, 3: 1}
    # M2: {C10: 1, C11: -1} -> {2: 1, 3: -1}
    # M3: {C01: 1, C11: 1} -> {1: 1, 3: 1}
    # M4: {C00: 1, C10: 1} -> {0: 1, 2: 1}
    # M5: {C00: -1, C01: 1} -> {0: -1, 1: 1}
    # M6: {C11: 1} -> {3: 1}
    # M7: {C00: 1} -> {0: 1}

    strassen_U = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1},      # M7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]

    col = 0
    for r in range(3):  # p-blocks (3 blocks of 2 columns)
        for s in range(2):  # m-blocks (2 blocks of 2 columns/rows)
            for t in range(7):  # Strassen terms
                for local_u, coeff in strassen_U[t].items():
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff

                for local_v, coeff in strassen_V[t].items():
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff

                for local_w, coeff in strassen_W[t].items():
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff

                col += 1

    if time.time() > deadline - 10.0:
        # Time pressure: return immediately without verification
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Exact verification using integer arithmetic
    try:
        U_int = U.astype(np.int64)
        V_int = V.astype(np.int64)
        W_int = W.astype(np.int64)

        # Check each tensor component T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = 0
                    for r_idx in range(rank):
                        s_val += U_int[i * m + j, r_idx] * V_int[j * p + k, r_idx] * W_int[k * n + i, r_idx]
                    if s_val != 1:
                        raise ValueError(f"Verification failed at ({i},{j},{k}): got {s_val}")
    except Exception:
        # Fallback to schoolbook
        r_sb = n * m * p
        U_sb = np.zeros((n * m, r_sb))
        V_sb = np.zeros((m * p, r_sb))
        W_sb = np.zeros((p * n, r_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
