import time
import numpy as np

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140

    # Strassen 2x2x2 (rank 7)
    A_strassen = [
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 1],
        [-1, 0],
        [0, 1]
    ]
    B_strassen = [
        [1, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, 0],
        [1, 1],
        [0, 1]
    ]
    C_strassen = [
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 0],
        [-1, 1],
        [0, 0],
        [1, 0]
    ]

    # Build 4x4x4 rank-49 via Strassen tiling
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))

    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            for b in range(2):
                for c in range(2):
                    a_coeff = A_strassen[t][b]
                    if a_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            a_micro = A_strassen[s][i]
                            if a_micro == 0:
                                continue
                            a_idx = (b * 2 + i) * 4 + (c * 2 + j)
                            U_444[a_idx, r] += a_coeff * a_micro

                    b_coeff = B_strassen[t][b]
                    if b_coeff == 0:
                        continue
                    for k in range(2):
                        for l in range(2):
                            b_micro = B_strassen[s][k]
                            if b_micro == 0:
                                continue
                            v_idx = (b * 2 + k) * 4 + (c * 2 + l)
                            V_444[v_idx, r] += b_coeff * b_micro

                    c_coeff = C_strassen[t][b]
                    if c_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            c_micro = C_strassen[s][i]
                            if c_micro == 0:
                                continue
                            w_row = b * 2 + i
                            w_col = c * 2 + j
                            w_idx = w_col * 4 + w_row
                            W_444[w_idx, r] += c_coeff * c_micro

    # Construct 4x4x8 by tiling the 4x4x4 decomposition
    # B is 4x8, split into B0 (4x4) and B1 (4x4)
    # C is 4x8, split into C0 (4x4) and C1 (4x4)
    # C0 = A * B0, C1 = A * B1
    rank_full = 98
    U_full = np.zeros((16, rank_full))
    V_full = np.zeros((32, rank_full))
    W_full = np.zeros((32, rank_full))

    # First half: multiply A by B0
    U_full[:, 0:49] = U_444
    V_full[0:16, 0:49] = V_444
    W_full[0:16, 0:49] = W_444

    # Second half: multiply A by B1
    U_full[:, 49:98] = U_444
    V_full[16:32, 49:98] = V_444
    W_full[16:32, 49:98] = W_444

    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    # Verify
    def verify(U_list, V_list, W_list, tol=1e-9):
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
