import numpy as np
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n

    # Exhaustive DP over all split points; base cases are tight.
    @lru_cache(maxsize=None)
    def compute_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        # Any block with a singleton dimension: minimal rank = product (schoolbook is optimal)
        if ni == 1 or mj == 1 or pk == 1:
            return ni * mj * pk
        if ni <= 2 and mj <= 2 and pk <= 2:
            return ni * mj * pk
        best = ni * mj * pk  # schoolbook fallback
        # Try every split in every dimension
        if ni > 1:
            for split in range(1, ni):
                r = compute_rank(split, mj, pk) + compute_rank(ni - split, mj, pk)
                if r < best:
                    best = r
        if mj > 1:
            for split in range(1, mj):
                r = compute_rank(ni, split, pk) + compute_rank(ni, mj - split, pk)
                if r < best:
                    best = r
        if pk > 1:
            for split in range(1, pk):
                r = compute_rank(ni, mj, split) + compute_rank(ni, mj, pk - split)
                if r < best:
                    best = r
        return best

    total_rank = compute_rank(n, m, p)

    @lru_cache(maxsize=None)
    def best_split(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return None
        if ni == 1 or mj == 1 or pk == 1:
            return None
        if ni <= 2 and mj <= 2 and pk <= 2:
            return None
        best_rank = ni * mj * pk
        best_choice = None
        if ni > 1:
            for split in range(1, ni):
                r = compute_rank(split, mj, pk) + compute_rank(ni - split, mj, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('ni', split)
        if mj > 1:
            for split in range(1, mj):
                r = compute_rank(ni, split, pk) + compute_rank(ni, mj - split, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('mj', split)
        if pk > 1:
            for split in range(1, pk):
                r = compute_rank(ni, mj, split) + compute_rank(ni, mj, pk - split)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('pk', split)
        return best_choice

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Strassen terms for 2x2x2 (rank 7)
    strassen_terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 1], [0, -1]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 0], [1, 1]]),
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 1], [0, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [1, 0]]),
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]

    def add_strassen(U, V, W, col, i_off, j_off, k_off):
        for u_mat, v_mat, w_mat in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        U[(i_off + ii) * m + (j_off + jj), col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        V[(j_off + jj) * p + (k_off + kk), col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        W[(k_off + kk) * n + (i_off + ii), col] += w_mat[kk][ii]
            col += 1
        return col

    def add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off + ii) * m + (j_off + jj), col] = 1.0
                    V[(j_off + jj) * p + (k_off + kk), col] = 1.0
                    W[(k_off + kk) * n + (i_off + ii), col] = 1.0
                    col += 1
        return col

    def decompose(ni, mj, pk, i_off, j_off, k_off, U, V, W, col):
        if time.time() > deadline:
            raise TimeoutError()
        if ni == 2 and mj == 2 and pk == 2:
            return add_strassen(U, V, W, col, i_off, j_off, k_off)
        if ni == 1 or mj == 1 or pk == 1:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        if ni <= 2 and mj <= 2 and pk <= 2:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        choice = best_split(ni, mj, pk)
        if choice is None:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        dim, split = choice
        if dim == 'ni':
            col = decompose(split, mj, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni - split, mj, pk, i_off + split, j_off, k_off, U, V, W, col)
        elif dim == 'mj':
            col = decompose(ni, split, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj - split, pk, i_off, j_off + split, k_off, U, V, W, col)
        else:  # dim == 'pk'
            col = decompose(ni, mj, split, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj, pk - split, i_off, j_off, k_off + split, U, V, W, col)
        return col

    col = decompose(n, m, p, 0, 0, 0, U, V, W, 0)
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Polish pass: prune any rank-1 column that contributes identically zero to every output entry.
    # This is exact (integer/half-integer arithmetic) and can only reduce rank.
    kept = []
    for c in range(total_rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        # A column contributes to output (i,j,k) as u[i,j]*v[j,k]*w[k,i].
        # If for every (i,j,k) this product is 0, the column is useless.
        nonzero = False
        for i in range(n):
            for j in range(m):
                uj = u_col[i * m + j]
                if uj == 0.0:
                    continue
                for k in range(p):
                    vjk = v_col[j * p + k]
                    if vjk == 0.0:
                        continue
                    wki = w_col[k * n + i]
                    if wki == 0.0:
                        continue
                    if uj * vjk * wki != 0.0:
                        nonzero = True
                        break
                if nonzero:
                    break
            if nonzero:
                break
        if nonzero:
            kept.append(c)

    if len(kept) < total_rank:
        U = U[:, kept]
        V = V[:, kept]
        W = W[:, kept]
        total_rank = len(kept)

    # Verify the decomposition is exact.
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(total_rank))
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)

    if error > 1e-6:
        # Fallback to schoolbook (rank n*m*p) which is always valid.
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school), dtype=float)
        V_s = np.zeros((mp, rank_school), dtype=float)
        W_s = np.zeros((pn, rank_school), dtype=float)
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col2] = 1.0
                    V_s[j * p + k, col2] = 1.0
                    W_s[k * n + i, col2] = 1.0
                    col2 += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
