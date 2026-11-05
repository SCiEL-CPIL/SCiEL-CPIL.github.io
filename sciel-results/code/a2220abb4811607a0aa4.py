"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Build a rank-44 base using 4 Strassen blocks (2x2x2) for the 2x2x8 part
     plus schoolbook for the 2x1x8 part.
  2. Search for linear dependencies among the 44 column vectors in the tensor
     space to attempt rank reduction.
  3. Use exact integer arithmetic for validation (all entries are integers).
  4. Return the best decomposition found within the time budget.
"""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10 s safety margin before 150 s kill

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

    # ---- Exact validation using integer arithmetic ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val == 0:
                            continue
                        v_val = v_row[c]
                        if v_val == 0:
                            continue
                        w_val = w_row[c]
                        if w_val == 0:
                            continue
                        acc += u_val * v_val * w_val
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
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Try to reduce rank via systematic column elimination ----
    def _try_reduce():
        """Attempt to find a smaller rank decomposition by eliminating columns."""
        nonlocal best_candidate, best_rank

        num_terms = n * m * p  # 48

        def _col_vector(U, V, W, c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            return vec

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        # Check for duplicate columns
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
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

        if not _elapsed_ok():
            return

        # Try to find pairs of columns that can be merged
        # If col_i + col_j = col_k, we have a dependency but can't directly reduce rank
        # However, we can try to find if removing one column and adjusting another works
        
        # More promising: check if any column is a scalar multiple of another
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                # Check if cols[i] = scalar * cols[j]
                # Find first nonzero in cols[i]
                found_i = False
                scalar = None
                for idx in range(num_terms):
                    if cols[i][idx] != 0:
                        if cols[j][idx] == 0:
                            found_i = False
                            break
                        # scalar = cols[i][idx] / cols[j][idx]
                        # Check if this scalar works for all
                        if scalar is None:
                            if cols[j][idx] == 0:
                                found_i = False
                                break
                            # Use integer division check
                            if cols[i][idx] % cols[j][idx] != 0:
                                found_i = False
                                break
                            scalar = cols[i][idx] // cols[j][idx]
                        else:
                            if cols[i][idx] != scalar * cols[j][idx]:
                                found_i = False
                                break
                        found_i = True
                    else:
                        if cols[j][idx] != 0:
                            found_i = False
                            break
                if found_i and scalar is not None:
                    # cols[i] = scalar * cols[j]
                    # We can potentially eliminate one of them
                    # Try removing col i
                    new_rank = rank0 - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    nc = 0
                    for c in range(rank0):
                        if c == i:
                            continue
                        for row in range(n * m):
                            new_U[row][nc] = U[row][c]
                        for row in range(m * p):
                            new_V[row][nc] = V[row][c]
                        for row in range(p * n):
                            new_W[row][nc] = W[row][c]
                        nc += 1
                    cand = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(cand):
                        best_candidate = cand
                        best_rank = new_rank
                        return

        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
