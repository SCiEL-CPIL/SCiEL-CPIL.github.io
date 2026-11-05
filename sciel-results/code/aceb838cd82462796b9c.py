"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Build a rank-44 decomposition using Strassen (7) for each 2x2 sub-block
     of the 2x2x8 part and schoolbook (16) for the 2x1x8 part.
  2. Use a fast integer-based elimination procedure (with NumPy for speed)
     to remove redundant terms.
  3. Also try two-term simultaneous elimination.
  4. Return the best (lowest-rank) valid decomposition found within the time budget.
"""
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback ----
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

    # ---- Validation ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        if not U or not V or not W:
            return False
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv != 0:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        # Check for any float entries that are not integers
        for mat in (U, V, W):
            for row in mat:
                for v in row:
                    if isinstance(v, float):
                        if v != int(v):
                            return False
        return True

    # ---- Build rank-44 Strassen + schoolbook decomposition ----
    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    # Verify Strassen internally
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

    rank_init = 44
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]

    col = 0

    # Part 1: A[0:2,0:2] * B[0:2,0:8] via 4 Strassen blocks
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A[0:2,2] * B[2,0:8] via schoolbook
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)

    def _try_eliminate_term_np(U_np, V_np, W_np, term_to_remove):
        """
        Try to express term `term_to_remove` as an integer linear combination
        of the other terms using numpy for speed.
        """
        rank = U_np.shape[1]
        keep_terms = [j for j in range(rank) if j != term_to_remove]
        num_keep = len(keep_terms)

        # Build the linear system using numpy
        # Number of equations: n*m*p = 48
        rhs = np.zeros(n * m * p, dtype=np.int64)
        A_mat = np.zeros((n * m * p, num_keep), dtype=np.int64)

        idx = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    rhs[idx] = U_np[i * m + a, term_to_remove] * V_np[a * p + k, term_to_remove] * W_np[k * n + i, term_to_remove]
                    for j_idx, j in enumerate(keep_terms):
                        A_mat[idx, j_idx] = U_np[i * m + a, j] * V_np[a * p + k, j] * W_np[k * n + i, j]
                    idx += 1

        # Use rational Gaussian elimination
        # Convert to Fraction for exact arithmetic
        from fractions import Fraction as Fr
        aug = [[Fr(0)] * (num_keep + 1) for _ in range(len(rhs))]
        for i in range(len(rhs)):
            for j in range(num_keep):
                aug[i][j] = Fr(A_mat[i, j])
            aug[i][num_keep] = Fr(rhs[i])

        # Gaussian elimination
        rank_aug = 0
        pivot_cols = []
        for col_idx in range(num_keep):
            pivot_row = -1
            for row_idx in range(rank_aug, len(rhs)):
                if aug[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
            pivot_cols.append(col_idx)
            pivot_val = aug[rank_aug][col_idx]
            for j in range(col_idx, num_keep + 1):
                aug[rank_aug][j] /= pivot_val
            for row_idx in range(len(rhs)):
                if row_idx == rank_aug:
                    continue
                factor = aug[row_idx][col_idx]
                if factor != 0:
                    for j in range(col_idx, num_keep + 1):
                        aug[row_idx][j] -= factor * aug[rank_aug][j]
            rank_aug += 1
            if not _elapsed_ok():
                return None, False

        # Check consistency
        for row_idx in range(rank_aug, len(rhs)):
            if aug[row_idx][num_keep] != 0:
                return None, False

        # Extract solution
        sol = [Fr(0)] * num_keep
        for pcol, pidx in enumerate(pivot_cols):
            sol[pidx] = aug[pcol][num_keep]

        # Check if solution is integer-valued
        for val in sol:
            if val != int(val):
                return None, False

        coeff_dict = {}
        for idx, j in enumerate(keep_terms):
            if sol[idx] != 0:
                coeff_dict[j] = int(sol[idx])

        return coeff_dict, True

    def _apply_elimination_np(U_np, V_np, W_np, term_to_remove, coeff_dict):
        rank = U_np.shape[1]
        new_U = []
        new_V = []
        new_W = []
        for j in range(rank):
            if j == term_to_remove:
                continue
            scale = 1 + coeff_dict.get(j, 0)
            if scale == 0:
                continue
            new_U.append([int(u) * scale for u in U_np[:, j]])
            new_V.append([int(v) * scale for v in V_np[:, j]])
            new_W.append([int(w) * scale for w in W_np[:, j]])
        return new_U, new_V, new_W

    def _try_eliminate_two_terms_np(U_np, V_np, W_np, term1, term2):
        rank = U_np.shape[1]
        keep_terms = [j for j in range(rank) if j != term1 and j != term2]
        num_keep = len(keep_terms)

        rhs = np.zeros(n * m * p, dtype=np.int64)
        A_mat = np.zeros((n * m * p, num_keep), dtype=np.int64)

        idx = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    rhs[idx] = (U_np[i * m + a, term1] * V_np[a * p + k, term1] * W_np[k * n + i, term1] +
                               U_np[i * m + a, term2] * V_np[a * p + k, term2] * W_np[k * n + i, term2])
                    for j_idx, j in enumerate(keep_terms):
                        A_mat[idx, j_idx] = U_np[i * m + a, j] * V_np[a * p + k, j] * W_np[k * n + i, j]
                    idx += 1

        from fractions import Fraction as Fr
        aug = [[Fr(0)] * (num_keep + 1) for _ in range(len(rhs))]
        for i in range(len(rhs)):
            for j in range(num_keep):
                aug[i][j] = Fr(A_mat[i, j])
            aug[i][num_keep] = Fr(rhs[i])

        rank_aug = 0
        pivot_cols = []
        for col_idx in range(num_keep):
            pivot_row = -1
            for row_idx in range(rank_aug, len(rhs)):
                if aug[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
            pivot_cols.append(col_idx)
            pivot_val = aug[rank_aug][col_idx]
            for j in range(col_idx, num_keep + 1):
                aug[rank_aug][j] /= pivot_val
            for row_idx in range(len(rhs)):
                if row_idx == rank_aug:
                    continue
                factor = aug[row_idx][col_idx]
                if factor != 0:
                    for j in range(col_idx, num_keep + 1):
                        aug[row_idx][j] -= factor * aug[rank_aug][j]
            rank_aug += 1
            if not _elapsed_ok():
                return None, False

        for row_idx in range(rank_aug, len(rhs)):
            if aug[row_idx][num_keep] != 0:
                return None, False

        sol = [Fr(0)] * num_keep
        for pcol, pidx in enumerate(pivot_cols):
            sol[pidx] = aug[pcol][num_keep]

        for val in sol:
            if val != int(val):
                return None, False

        coeff_dict = {}
        for idx, j in enumerate(keep_terms):
            if sol[idx] != 0:
                coeff_dict[j] = int(sol[idx])

        return coeff_dict, True

    def _apply_two_elimination_np(U_np, V_np, W_np, term1, term2, coeff_dict):
        rank = U_np.shape[1]
        new_U = []
        new_V = []
        new_W = []
        for j in range(rank):
            if j == term1 or j == term2:
                continue
            scale = 1 + coeff_dict.get(j, 0)
            if scale == 0:
                continue
            new_U.append([int(u) * scale for u in U_np[:, j]])
            new_V.append([int(v) * scale for v in V_np[:, j]])
            new_W.append([int(w) * scale for w in W_np[:, j]])
        return new_U, new_V, new_W

    # Greedy elimination loop
    best_U, best_V, best_W = U, V, W
    best_rank = rank_init

    # Phase 1: Single-term elimination
    progress = True
    while progress and _elapsed_ok():
        progress = False
        current_rank = len(best_U[0]) if best_U else 0
        best_U_np = np.array(best_U, dtype=np.int64)
        best_V_np = np.array(best_V, dtype=np.int64)
        best_W_np = np.array(best_W, dtype=np.int64)
        for r in range(current_rank):
            if not _elapsed_ok():
                break
            coeff_dict, success = _try_eliminate_term_np(best_U_np, best_V_np, best_W_np, r)
            if success:
                new_U, new_V, new_W = _apply_elimination_np(best_U_np, best_V_np, best_W_np, r, coeff_dict)
                new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(new_candidate):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(new_U[0]) if new_U else 0
                    progress = True
                    if not _elapsed_ok():
                        break

    # Phase 2: Two-term elimination
    if _elapsed_ok() and best_rank > 40:
        progress = True
        while progress and _elapsed_ok():
            progress = False
            current_rank = len(best_U[0]) if best_U else 0
            best_U_np = np.array(best_U, dtype=np.int64)
            best_V_np = np.array(best_V, dtype=np.int64)
            best_W_np = np.array(best_W, dtype=np.int64)
            for r1 in range(current_rank):
                for r2 in range(r1 + 1, current_rank):
                    if not _elapsed_ok():
                        break
                    coeff_dict, success = _try_eliminate_two_terms_np(best_U_np, best_V_np, best_W_np, r1, r2)
                    if success:
                        new_U, new_V, new_W = _apply_two_elimination_np(best_U_np, best_V_np, best_W_np, r1, r2, coeff_dict)
                        new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(new_candidate):
                            best_U, best_V, best_W = new_U, new_V, new_W
                            best_rank = len(new_U[0]) if new_U else 0
                            progress = True
                            if not _elapsed_ok():
                                break
                if not _elapsed_ok():
                    break

    # Phase 3: Final single-term pass
    if _elapsed_ok() and best_rank > 40:
        progress = True
        while progress and _elapsed_ok():
            progress = False
            current_rank = len(best_U[0]) if best_U else 0
            best_U_np = np.array(best_U, dtype=np.int64)
            best_V_np = np.array(best_V, dtype=np.int64)
            best_W_np = np.array(best_W, dtype=np.int64)
            for r in range(current_rank):
                if not _elapsed_ok():
                    break
                coeff_dict, success = _try_eliminate_term_np(best_U_np, best_V_np, best_W_np, r)
                if success:
                    new_U, new_V, new_W = _apply_elimination_np(best_U_np, best_V_np, best_W_np, r, coeff_dict)
                    new_candidate = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(new_candidate):
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = len(new_U[0]) if new_U else 0
                        progress = True
                        if not _elapsed_ok():
                            break

    if best_rank < rank_init:
        result = {"U": best_U, "V": best_V, "W": best_W}
        if _validate(result):
            return result

    return candidate
