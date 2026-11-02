"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Uses a structured decomposition that exploits the n=2 structure to share
intermediate products across column blocks of B, aiming for rank < 44.

Strategy:
- Split the 8-column dimension into 4 blocks of 2 columns each.
- For each block, split the 3-row dimension into 1 row + 2 rows.
- The 1-row part (2x1 * 1x2) has rank 4 (outer product).
- The 2-row part (2x2 * 2x2) uses Strassen's algorithm with rank 7.
- Total rank per block: 4 + 7 = 11.
- Total rank: 4 * 11 = 44.

Index conventions:
- U has shape (n*m, rank) = (6, rank): index a = i*m + j
- V has shape (m*p, rank) = (24, rank): index b = j*p + k
- W has shape (p*n, rank) = (16, rank): index c = k*n + i
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _get_target_indices():
    n, m, p = DIMENSIONS
    target = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = True
    return target


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])

    target = _get_target_indices()
    target_err = np.max(np.abs(T[target] - 1.0))
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    return max(target_err, non_target_err) <= tol


def _build_rank44():
    """Explicit rank-44 decomposition: 4 blocks of (7 Strassen + 4 outer)."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for pair in range(4):
        k0 = 2 * pair
        k1 = 2 * pair + 1

        # Outer product terms for A[:,0] * B[0,k0:k1]
        # t0 = A[0,0] * B[0,k0] -> C[0,k0]
        U[0, col] = 1
        V[0 * p + k0, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

        # t1 = A[0,0] * B[0,k1] -> C[0,k1]
        U[0, col] = 1
        V[0 * p + k1, col] = 1
        W[k1 * n + 0, col] = 1
        col += 1

        # t2 = A[1,0] * B[0,k0] -> C[1,k0]
        U[1 * m + 0, col] = 1
        V[0 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1

        # t3 = A[1,0] * B[0,k1] -> C[1,k1]
        U[1 * m + 0, col] = 1
        V[0 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # Strassen terms for A[:,1:3] * B[1:3,k0:k1]
        # M1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # M2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1

        # M3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1, col] = 1
        V[1 * p + k1, col] = 1
        V[2 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # M4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        V[1 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1

        # M5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1, col] = 1
        U[0 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1

        # M6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1, col] = 1
        U[0 * m + 1, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # M7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = -1
        V[2 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank_reduction(U, V, W, target_rank, deadline):
    """Try to reduce rank by removing terms."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]

    if current_rank <= target_rank:
        return None

    target = _get_target_indices()

    for r in range(current_rank):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_rank) if i != r]
        U_sub = U_arr[:, mask]
        V_sub = V_arr[:, mask]
        W_sub = W_arr[:, mask]
        T_sub = np.zeros((n * m, m * p, p * n))
        for rr in range(len(mask)):
            T_sub += np.einsum('a,b,c->abc', U_sub[:, rr], V_sub[:, rr], W_sub[:, rr])
        target_err = np.max(np.abs(T_sub[target] - 1.0))
        non_target_err = np.max(np.abs(T_sub[~target])) if (~target).any() else 0.0
        if max(target_err, non_target_err) < 1e-9:
            return U_sub.tolist(), V_sub.tolist(), W_sub.tolist()

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the explicit rank-44 decomposition
    U44, V44, W44 = _build_rank44()

    if not _verify(U44, V44, W44, tol=1e-9):
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44

    # Try to reduce rank below 44
    if time.time() < deadline - 5:
        reduced = _try_rank_reduction(U44, V44, W44, 43, deadline)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify(U_r, V_r, W_r, tol=1e-9):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = len(U_r[0])

    return {"U": best_U, "V": best_V, "W": best_W}
