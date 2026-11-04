"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via local search with exact rational arithmetic.
     - Try removing redundant columns
     - Try merging pairs of columns via linear combination
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the known rank-44 decomposition.
"""
import time
import sys
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # 15 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                W_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[W_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Helper: compute residual tensor when excluding a set of columns ----
    def _residual_excluding(U, V, W, rank, exclude_set):
        """Compute residual[i][a][k] = 1 - sum_{c not in exclude_set} U[i*m+a][c]*V[a*p+k][c]*W[k*n+i][c]"""
        residual = [[[0]*p for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    for c in range(rank):
                        if c in exclude_set:
                            continue
                        u_val = U[i * m + a][c]
                        v_val = V[a * p + k][c]
                        w_val = W[k * n + i][c]
                        acc += u_val * v_val * w_val
                    residual[i][a][k] = 1 - acc
        return residual

    def _is_zero_residual(residual):
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    if residual[i][a][k] != 0:
                        return False
        return True

    # ---- Try removing each column and checking if we can compensate ----
    def _try_remove_column():
        nonlocal best_candidate, best_rank

        if not _elapsed_ok():
            return

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        for remove_idx in range(rank0):
            if not _elapsed_ok():
                return

            new_rank = rank0 - 1

            # Check if removing this column still gives correct result
            # i.e., the column was redundant (all entries zero in the tensor)
            is_redundant = True
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        # If U[i*m+a][remove_idx] * V[a*p+k][remove_idx] * W[k*n+i][remove_idx] != 0
                        # then this column contributes to the tensor
                        if U[i * m + a][remove_idx] != 0 and V[a * p + k][remove_idx] != 0 and W[k * n + i][remove_idx] != 0:
                            is_redundant = False
                            break
                    if not is_redundant:
                        break
                if not is_redundant:
                    break
            if not is_redundant:
                continue

            # Build new candidate without this column
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            nc = 0
            for c in range(rank0):
                if c == remove_idx:
                    continue
                for i in range(n * m):
                    new_U[i][nc] = U[i][c]
                for i in range(m * p):
                    new_V[i][nc] = V[i][c]
                for i in range(p * n):
                    new_W[i][nc] = W[i][c]
                nc += 1

            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                best_candidate = cand
                best_rank = new_rank
                return

    # ---- Try merging pairs of columns via linear combination ----
    def _try_merge_columns():
        nonlocal best_candidate, best_rank

        if not _elapsed_ok():
            return

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        # For each pair of columns (c1, c2), try to find alpha, beta such that
        # replacing both with a single column alpha*col1 + beta*col2 still works
        # This is a 2x2 system in the tensor space
        for c1 in range(rank0):
            if not _elapsed_ok():
                return
            for c2 in range(c1 + 1, rank0):
                if not _elapsed_ok():
                    return

                # Check if columns c1 and c2 have the same support pattern
                # (i.e., for each (i,a,k), either both contribute or neither does)
                same_support = True
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            u1 = U[i * m + a][c1]
                            v1 = V[a * p + k][c1]
                            w1 = W[k * n + i][c1]
                            u2 = U[i * m + a][c2]
                            v2 = V[a * p + k][c2]
                            w2 = W[k * n + i][c2]
                            prod1 = (u1 != 0 and v1 != 0 and w1 != 0)
                            prod2 = (u2 != 0 and v2 != 0 and w2 != 0)
                            if prod1 != prod2:
                                same_support = False
                                break
                        if not same_support:
                            break
                    if not same_support:
                        break
                if not same_support:
                    continue

                # Try simple linear combinations: alpha=1, beta=0 (just c1) or alpha=0, beta=1 (just c2)
                # These are covered by _try_remove_column, so try alpha=1, beta=1
                # and alpha=1, beta=-1, etc.
                for alpha, beta in [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 1), (1, 2), (2, -1), (-1, 2)]:
                    if not _elapsed_ok():
                        return

                    new_rank = rank0 - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    nc = 0
                    for c in range(rank0):
                        if c == c1 or c == c2:
                            continue
                        for i in range(n * m):
                            new_U[i][nc] = U[i][c]
                        for i in range(m * p):
                            new_V[i][nc] = V[i][c]
                        for i in range(p * n):
                            new_W[i][nc] = W[i][c]
                        nc += 1

                    # Add the merged column
                    for i in range(n * m):
                        new_U[i][nc] = alpha * U[i][c1] + beta * U[i][c2]
                    for i in range(m * p):
                        new_V[i][nc] = alpha * V[i][c1] + beta * V[i][c2]
                    for i in range(p * n):
                        new_W[i][nc] = alpha * W[i][c1] + beta * W[i][c2]

                    cand = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(cand):
                        best_candidate = cand
                        best_rank = new_rank
                        return

    # ---- Run reduction attempts ----
    if _elapsed_ok():
        _try_remove_column()

    if _elapsed_ok() and best_rank == 44:
        _try_merge_columns()

    return best_candidate

# EVOLVE-BLOCK-END
