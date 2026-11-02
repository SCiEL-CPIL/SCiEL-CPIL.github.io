"""Matrix multiplication 2,2,15 tensor rank minimization.

Strategy:
  * Base: 7 independent Strassen blocks for column pairs (0,1)..(12,13) plus a
    rank-4 schoolbook block for column 14 -> rank 53.
  * Try to reduce to rank 52 by finding a rank-10 decomposition for 2x2x3 and
    using 6 Strassen blocks + 1 rank-10 block.
  * Additionally try a "sliding-window" overlap: treat columns as a sequence of
    overlapping 3-column windows (0-2, 3-5, ..., 12-14) each with a rank-10
    2x2x3 decomposition, but allow the windows to share the middle column's
    contributions so that the total rank is less than 5 * 10 = 50.  In practice
    we attempt to find a rank-9 or rank-10 2x2x3 decomposition and combine
    windows with shared rank-1 terms for the overlapping column.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # seconds; evaluator kills at 150s


def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1

    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1

    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1

    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_2x2x3_rank11():
    """
    Build a rank-11 decomposition for 2x2 x 2x3 matrix multiplication.
    Uses 7 Strassen terms for columns 0-1 and 4 schoolbook terms for column 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U_str, V_str, W_str = _build_strassen_2x2x2()

    # First 7 terms: Strassen for columns 0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local * n + i, 0:7] = W_str[k_local * 2 + i, :]

    # Terms 7-10: Schoolbook for column 2
    U[0, 7] = 1; V[0, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[2, 9] = 1; V[4, 9] = 1; W[5, 9] = 1
    U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1

    return U, V, W


def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)

    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])

        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _find_2x2x3_rank10(deadline):
    """
    Search for a rank-10 decomposition of 2x2x3 matrix multiplication.
    Returns (U, V, W) of shapes (4,10), (6,10), (6,10) or None.
    """
    n, m, p = 2, 2, 3
    rank = 10

    np.random.seed(42)
    for attempt in range(50000):
        if time.time() > deadline:
            break
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))

        for r in range(rank):
            n_u = np.random.randint(1, 4)
            u_rows = np.random.choice(4, n_u, replace=False)
            for row in u_rows:
                U[row, r] = np.random.choice([-1, 1])

            n_v = np.random.randint(1, 4)
            v_rows = np.random.choice(6, n_v, replace=False)
            for row in v_rows:
                V[row, r] = np.random.choice([-1, 1])

            n_w = np.random.randint(1, 3)
            w_rows = np.random.choice(6, n_w, replace=False)
            for row in w_rows:
                W[row, r] = np.random.choice([-1, 1])

        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return U, V, W

    return None


def _try_sliding_window(deadline):
    """
    Attempt a sliding-window decomposition:
    5 windows of 3 columns each: (0,1,2), (3,4,5), ..., (12,13,14).
    If we can find a rank-10 2x2x3 decomposition, total rank = 5*10 = 50.
    But we can potentially do better by sharing the middle column between
    adjacent windows.  For now, just try 5 independent rank-10 blocks.
    If that gives rank 50, great.  If not, try rank-11 blocks -> 55 (worse than 53).
    """
    n, m, p = DIMENSIONS
    result = _find_2x2x3_rank10(deadline)
    if result is None:
        return None
    U3, V3, W3 = result
    total_rank = 5 * 10
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for w in range(5):
        col_offset = w * 10
        start_k = w * 3
        U[:, col_offset:col_offset+10] = U3
        for j in range(m):
            for k_local in range(3):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V3[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W3[k_local * 2 + i, :]

    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    return None


def _build_rank52(deadline):
    """
    Build a rank-52 decomposition if possible, otherwise fall back to rank-53.
    Uses 6 Strassen blocks (cols 0-11) + 1 rank-10 block (cols 12-14).
    """
    n, m, p = DIMENSIONS

    result = _find_2x2x3_rank10(deadline)
    if result is not None:
        U3, V3, W3 = result
        U_str, V_str, W_str = _build_strassen_2x2x2()
        total_rank = 6 * 7 + 10  # 52
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))

        col_offset = 0
        for b in range(6):
            start_k = b * 2
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7

        # Last block: 2x2x3 rank-10 for cols 12,13,14
        U[:, col_offset:col_offset+10] = U3
        for j in range(m):
            for k_local in range(3):
                k_global = 12 + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V3[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = 12 + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W3[k_local * 2 + i, :]

        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W

    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    n, m, p = DIMENSIONS

    # Try rank 50 via sliding window first (best possible if found)
    result = _try_sliding_window(deadline)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 52
    result = _build_rank52(deadline)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 53
    U, V, W = _build_rank53()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
