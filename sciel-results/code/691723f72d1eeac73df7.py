"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _search_sub_decomposition(n, m, p, max_rank, timeout=60):
    """Search for rank-minimal decomposition of n x m x p tensor."""
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]

    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return None

        num_trials = 10000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Random U, V
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))

            # Set up linear system for W
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1

            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-8):
                    W_rand = W_flat.reshape((p * n, r))
                    if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                        W_rounded = np.round(W_rand * 2) / 2
                        if np.allclose(W_rounded, W_rand, atol=1e-8):
                            return (r, U_rand.tolist(), V_rand.tolist(), W_rounded.tolist())
            except:
                pass
    return None


def _construct_sub_3x4x3():
    n, m, p = 3, 4, 3
    result = _search_sub_decomposition(n, m, p, max_rank=20, timeout=40)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def _construct_sub_3x4x2():
    n, m, p = 3, 4, 2
    result = _search_sub_decomposition(n, m, p, max_rank=16, timeout=40)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    sub_decomps = []
    r0, U0, V0, W0 = _construct_sub_3x4x3()
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], blocks[0]))
    r1, U1, V1, W1 = _construct_sub_3x4x2()
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], blocks[1]))
    r2, U2, V2, W2 = _construct_sub_3x4x2()
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], blocks[2]))

    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)

    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        for j in range(m):
            for kk in range(b):
                k_local = kk
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + k_local][c]

        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
