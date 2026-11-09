import time
import numpy as np

def construct():
    """
    Return a rank-92 exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses Strassen 7-term decomposition on 8 non-overlapping 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20

    # 8 Strassen blocks: i in {0,1} or {2,3}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    # Each block saves 1 term vs standard 8-term: total saving = 8
    # Rank = 100 - 8 = 92
    rank = 92
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # Define the 8 Strassen blocks
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((i_base, j_base, k_base))

    # Set of (i,j,k) covered by Strassen blocks
    skip_set = set()
    for ib, jb, kb in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    skip_set.add((ib + di, jb + dj, kb + dk))

    col = 0
    # Fill standard terms for non-block entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Fill Strassen terms for each 2x2x2 block
    for ib, jb, kb in blocks:
        i1, i2 = ib, ib + 1
        j1, j2 = jb, jb + 1
        k1, k2 = kb, kb + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[i2 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i2, col] += 1
        W[k2 * n + i2, col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[i1 * m + j1, col] += 1
        V[j1 * p + k2, col] += 1
        V[j2 * p + k2, col] += -1
        W[k2 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[i2 * m + j2, col] += 1
        V[j2 * p + k1, col] += 1
        V[j1 * p + k1, col] += -1
        W[k1 * n + i1, col] += 1
        W[k1 * n + i2, col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[i1 * m + j1, col] += 1
        U[i1 * m + j2, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += -1
        W[k2 * n + i1, col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1, col] += 1
        U[i1 * m + j1, col] += -1
        V[j1 * p + k1, col] += 1
        V[j1 * p + k2, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2, col] += 1
        U[i2 * m + j2, col] += -1
        V[j2 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1

    # Verify correctness within time budget (vectorized)
    if time.time() < deadline - 5:
        # Reshape for vectorized check
        U_r = U  # shape (nm, rank)
        V_r = V  # shape (mp, rank)
        W_r = W  # shape (pn, rank)
        
        all_ok = True
        for test_i in range(n):
            for test_j in range(m):
                u_row = U_r[test_i * m + test_j, :]  # shape (rank,)
                for test_k in range(p):
                    v_row = V_r[test_j * p + test_k, :]  # shape (rank,)
                    w_row = W_r[test_k * n + test_i, :]  # shape (rank,)
                    actual = np.dot(u_row, v_row * w_row)
                    if actual != 1:
                        all_ok = False
                        break
                if not all_ok:
                    break
            if not all_ok:
                break
        
        if not all_ok:
            # Fallback: return standard rank-100 decomposition
            rank_std = n * m * p
            U_std = np.zeros((nm, rank_std), dtype=np.int64)
            V_std = np.zeros((mp, rank_std), dtype=np.int64)
            W_std = np.zeros((pn, rank_std), dtype=np.int64)
            col_std = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_std[i * m + j, col_std] = 1
                        V_std[j * p + k, col_std] = 1
                        W_std[k * n + i, col_std] = 1
                        col_std += 1
            return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
