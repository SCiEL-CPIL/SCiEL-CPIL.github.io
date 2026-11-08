"""Strassen-based hybrid decomposition for 2x3 times 3x8 matrix multiplication.
Uses Strassen for 2x2 blocks and schoolbook for remaining parts, with a search
for additional rank reduction via algebraic manipulation."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

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
        import numpy as np
        U = np.array(candidate["U"], dtype=np.int64)
        V = np.array(candidate["V"], dtype=np.int64)
        W = np.array(candidate["W"], dtype=np.int64)
        if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]) or W.shape != (p * n, W.shape[1]):
            return False
        rank = U.shape[1]
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        try:
            T = np.einsum('iac,akc,kic->iak', U_t, V_t, W_t, optimize=True)
        except Exception:
            return False
        if not np.all(T == 1):
            return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
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

    # Build base candidate with rank 44
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A[0:2, 0:2] * B[0:2, :] using Strassen on 2x2x2 blocks
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
                if not _elapsed_ok():
                    return _schoolbook()

        # Part 2: A[0:2, 2] * B[2, :] using schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                if not _elapsed_ok():
                    return _schoolbook()

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if _validate(base):
        # Try to reduce rank by finding linear dependencies
        import numpy as np
        U = np.array(base["U"], dtype=np.int64)
        V = np.array(base["V"], dtype=np.int64)
        W = np.array(base["W"], dtype=np.int64)
        rank = U.shape[1]

        # Attempt rank reduction by checking if any column is a linear combination
        # of others, using integer arithmetic
        best_rank = rank
        best_candidate = base

        # Try removing pairs of columns and checking if the tensor is still valid
        # This is a heuristic search for reducibility
        for trial in range(min(rank, 10)):
            if not _elapsed_ok():
                break

            # Try to find a column that can be expressed as a combination of others
            # Use a simple approach: try removing column c and solving for its coefficients
            for c in range(rank):
                if not _elapsed_ok():
                    break

                # Skip if already tried
                if c >= rank:
                    continue

                # Try removing column c
                new_rank = rank - 1
                if new_rank >= best_rank:
                    continue

                U_new = np.delete(U, c, axis=1)
                V_new = np.delete(V, c, axis=1)
                W_new = np.delete(W, c, axis=1)

                # Check if the tensor is still all ones
                U_t = U_new.reshape(n, m, new_rank)
                V_t = V_new.reshape(m, p, new_rank)
                W_t = W_new.reshape(p, n, new_rank)

                try:
                    T = np.einsum('iac,akc,kic->iak', U_t, V_t, W_t, optimize=True)
                except Exception:
                    continue

                if np.all(T == 1):
                    best_rank = new_rank
                    best_candidate = {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
                    rank = new_rank
                    U, V, W = U_new, V_new, W_new
                    break

        if _validate(best_candidate):
            return best_candidate
        return base
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
