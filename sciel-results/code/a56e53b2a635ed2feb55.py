"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    import numpy as np

    # 2x2x2 Strassen rank-7 decomposition
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ], dtype=float)

    def verify(U_arr, V_arr, W_arr, n, m, p):
        """Verify tensor decomposition."""
        rank = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # Vectorized: T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
            # Reshape and use einsum-like approach
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_fast(U_arr, V_arr, W_arr, n, m, p):
        """Faster verification using matrix operations."""
        rank = U_arr.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For each r, outer product of reshaped vectors
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(T, 1.0, atol=1e-9)

    # Build 4x4x4 Strassen decomposition (rank 49)
    # Hierarchical: block-level Strassen (7 terms) x sub-block Strassen (7 terms) = 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))

    for bt in range(7):
        for st in range(7):
            col = bt * 7 + st
            for ib in range(2):
                for ii in range(2):
                    i = ib * 2 + ii
                    for jb in range(2):
                        for jj in range(2):
                            j = jb * 2 + jj
                            U_444[i * 4 + j, col] = S_U[ib * 2 + jb, bt] * S_U[ii * 2 + jj, st]
            for jb in range(2):
                for jj in range(2):
                    j = jb * 2 + jj
                    for kb in range(2):
                        for kk in range(2):
                            k = kb * 2 + kk
                            V_444[j * 4 + k, col] = S_V[jb * 2 + kk, bt] * S_V[jj * 2 + kk, st]
            for kb in range(2):
                for kk in range(2):
                    k = kb * 2 + kk
                    for ib in range(2):
                        for ii in range(2):
                            i = ib * 2 + ii
                            W_444[k * 4 + i, col] = S_W[kb * 2 + ii, bt] * S_W[kk * 2 + ii, st]

    # Verify 4x4x4 part in isolation
    if time.time() < deadline - 60:
        if not verify_fast(U_444, V_444, W_444, 4, 4, 4):
            # Fallback to simple block-wise
            pass  # We'll use fallback below

    # Build 4x4x2 part (rank 28)
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((8, 28))
    W_442 = np.zeros((8, 28))

    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = col_offset + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U_442[i * 4 + j, col] = S_U[di * 2 + dj, c]
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = dk
                        V_442[j * 2 + k, col] = S_V[dj * 2 + dk, c]
                for dk in range(2):
                    for di in range(2):
                        k = dk
                        i = bi * 2 + di
                        W_442[k * 4 + i, col] = S_W[dk * 2 + di, c]
            col_offset += 7

    # Combine into full 4x4x6
    total_rank = 77
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))

    # 4x4x4 part (k=0..3)
    U[:, :49] = U_444
    V[:16, :49] = V_444
    W[:16, :49] = W_444

    # 4x4x2 part (k=4,5)
    U[:, 49:] = U_442
    V[16:, 49:] = V_442
    W[16:, 49:] = W_442

    # Verify full decomposition
    if time.time() < deadline - 30:
        if verify_fast(U, V, W, n, m, p):
            # Try to remove redundant columns
            removed = set()
            for r in range(total_rank):
                if time.time() > deadline - 10:
                    break
                if r in removed:
                    continue
                U_trial = U.copy()
                V_trial = V.copy()
                W_trial = W.copy()
                U_trial[:, r] = 0
                V_trial[:, r] = 0
                W_trial[:, r] = 0
                if verify_fast(U_trial, V_trial, W_trial, n, m, p):
                    removed.add(r)
                    U = U_trial
                    V = V_trial
                    W = W_trial

            if removed:
                keep_cols = [c for c in range(total_rank) if c not in removed]
                U = U[:, keep_cols]
                V = V[:, keep_cols]
                W = W[:, keep_cols]

            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: simple 2x2x2 block-wise Strassen (rank 84)
    n_blocks = n // 2
    m_blocks = m // 2
    p_blocks = p // 2
    total_rank_fb = n_blocks * m_blocks * p_blocks * 7
    U_fb = np.zeros((n * m, total_rank_fb))
    V_fb = np.zeros((m * p, total_rank_fb))
    W_fb = np.zeros((p * n, total_rank_fb))
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U_fb[i * m + j, col] = S_U[di * 2 + dj, c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V_fb[j * p + k, col] = S_V[dj * 2 + dk, c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W_fb[k * n + i, col] = S_W[dk * 2 + di, c]
                col_offset += 7

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
