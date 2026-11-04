"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

We build a 44-term decomposition (Strassen on 2x2 sub-blocks for the first two
columns of A, plus schoolbook for the third column) and then attempt to reduce
the rank by identifying and removing linearly dependent (redundant) terms,
while preserving exactness over the integers/rationals.
"""

import time
import sys
from fractions import Fraction


def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- helpers ---------------------------------------------------------

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
                        u_val = u_row[c]
                        v_val = v_row[c]
                        w_val = w_row[c]
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            continue
                        acc += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 coefficients ------------------------------------

    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- build 44-term decomposition -------------------------------------

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    best_rank = 44
    best_candidate = {"U": U, "V": V, "W": W}

    if not _validate(best_candidate):
        return _schoolbook()

    # ---- attempt to reduce rank by removing redundant terms --------------

    def _try_reduce(candidate):
        """Try to reduce rank by removing linearly dependent terms."""
        U_c = candidate["U"]
        V_c = candidate["V"]
        W_c = candidate["W"]
        r = len(U_c[0])
        best_r = r
        best_c = None

        # Build the list of non-zero term indices (all should be non-zero initially)
        for trial in range(r):
            if not _elapsed_ok():
                break
            # Try removing term `trial`
            new_U = [row[:trial] + row[trial + 1:] for row in U_c]
            new_V = [row[:trial] + row[trial + 1:] for row in V_c]
            new_W = [row[:trial] + row[trial + 1:] for row in W_c]
            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                best_r = trial
                best_c = cand
                # Reassign U_c, V_c, W_c to the reduced ones
                U_c, V_c, W_c = new_U, new_V, new_W
                r = trial
                # Restart trial loop since indices shifted
                trial = 0
            else:
                trial += 1

        if best_c is not None:
            return best_r, best_c
        return best_r, candidate

    if _elapsed_ok():
        new_rank, new_candidate = _try_reduce(best_candidate)
        if new_rank < best_rank:
            best_rank = new_rank
            best_candidate = new_candidate
            print(f"Reduced rank to {best_rank}", file=sys.stderr)

    if not _elapsed_ok():
        return best_candidate

    return best_candidate
