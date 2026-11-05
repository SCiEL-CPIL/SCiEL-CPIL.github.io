"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from functools import lru_cache

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    # Strassen terms for 2x2x2: (U_mat, V_mat, W_mat)
    # U_mat: 2x2, V_mat: 2x2, W_mat: 2x2
    # Convention: U[i,j], V[j,k], W[k,i]
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]

    # Compute minimum rank using DP over all possible splits
    @lru_cache(maxsize=None)
    def min_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        if ni <= 2 and mj <= 2 and pk <= 2:
            return ni * mj * pk
        best = ni * mj * pk  # schoolbook
        # Try splitting ni
        for split in range(1, ni):
            best = min(best, min_rank(split, mj, pk) + min_rank(ni-split, mj, pk))
        # Try splitting mj
        for split in range(1, mj):
            best = min(best, min_rank(ni, split, pk) + min_rank(ni, mj-split, pk))
        # Try splitting pk
        for split in range(1, pk):
            best = min(best, min_rank(ni, mj, split) + min_rank(ni, mj, pk-split))
        return best

    total_rank = min_rank(n, m, p)

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    def add_strassen_block(U, V, W, col, i_off, j_off, k_off):
        for u_mat, v_mat, w_mat in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    val = u_mat[ii][jj]
                    if val != 0:
                        U[(i_off+ii)*m + (j_off+jj), col] += val
            for jj in range(2):
                for kk in range(2):
                    val = v_mat[jj][kk]
                    if val != 0:
                        V[(j_off+jj)*p + (k_off+kk), col] += val
            for kk in range(2):
                for ii in range(2):
                    val = w_mat[kk][ii]
                    if val != 0:
                        W[(k_off+kk)*n + (i_off+ii), col] += val
            col += 1
        return col

    def add_schoolbook_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off+ii)*m + (j_off+jj), col] = 1
                    V[(j_off+jj)*p + (k_off+kk), col] = 1
                    W[(k_off+kk)*n + (i_off+ii), col] = 1
                    col += 1
        return col

    def build(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return add_strassen_block(U, V, W, col, i_off, j_off, k_off)
        if ni <= 2 and mj <= 2 and pk <= 2:
            return add_schoolbook_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)

        best = ni * mj * pk
        best_split = None

        # Try splitting ni
        for split in range(1, ni):
            cost = min_rank(split, mj, pk) + min_rank(ni-split, mj, pk)
            if cost < best:
                best = cost
                best_split = ('i', split)
        # Try splitting mj
        for split in range(1, mj):
            cost = min_rank(ni, split, pk) + min_rank(ni, mj-split, pk)
            if cost < best:
                best = cost
                best_split = ('j', split)
        # Try splitting pk
        for split in range(1, pk):
            cost = min_rank(ni, mj, split) + min_rank(ni, mj, pk-split)
            if cost < best:
                best = cost
                best_split = ('k', split)

        if best_split is None:
            return add_schoolbook_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)

        axis, split = best_split
        if axis == 'i':
            col = build(U, V, W, col, i_off, j_off, k_off, split, mj, pk)
            col = build(U, V, W, col, i_off+split, j_off, k_off, ni-split, mj, pk)
        elif axis == 'j':
            col = build(U, V, W, col, i_off, j_off, k_off, ni, split, pk)
            col = build(U, V, W, col, i_off, j_off+split, k_off, ni, mj-split, pk)
        else:
            col = build(U, V, W, col, i_off, j_off, k_off, ni, mj, split)
            col = build(U, V, W, col, i_off, j_off, k_off+split, ni, mj, pk-split)
        return col

    col = build(U, V, W, 0, 0, 0, n, m, p)

    if col != total_rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        total_rank = col

    # Verify the decomposition
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    rank = total_rank

    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                if s != 1:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if not verified:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
