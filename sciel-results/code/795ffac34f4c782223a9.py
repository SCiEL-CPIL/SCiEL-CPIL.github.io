import time
import numpy as np

DIMENSIONS = [4, 4, 9]
N, M, P = 4, 4, 9
DEADLINE_MARGIN = 20.0


def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W


def _build_block_decomp():
    """Build the 114-rank block decomposition (two Strassen 4x4x4 + schoolbook 4x4x1)."""
    U2, V2, W2 = _strassen_2x2()
    U4_44 = np.kron(U2, U2)
    V4_44 = np.kron(V2, V2)
    W4_44 = np.kron(W2, W2)

    # Schoolbook 4x4x1
    U4_16 = np.zeros((16, 16))
    V4_16 = np.zeros((4, 16))
    W4_16 = np.zeros((4, 16))
    col = 0
    for i in range(4):
        for j in range(4):
            U4_16[i * 4 + j, col] = 1.0
            V4_16[j, col] = 1.0
            W4_16[i, col] = 1.0
            col += 1

    total_rank = 114
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))

    # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C)
    U_full[:, 0:49] = U4_44
    for r_v in range(16):
        j = r_v // 4
        k = r_v % 4
        V_full[j * 9 + k, 0:49] = V4_44[r_v, :]
    for r_w in range(16):
        k = r_w // 4
        i = r_w % 4
        W_full[k * 4 + i, 0:49] = W4_44[r_w, :]

    # Part 2: 4x4x4 (cols 4-7 of B, rows 4-7 of C)
    offset = 49
    U_full[:, offset:offset + 49] = U4_44
    for r_v in range(16):
        j = r_v // 4
        k_local = r_v % 4
        k_global = k_local + 4
        V_full[j * 9 + k_global, offset:offset + 49] = V4_44[r_v, :]
    for r_w in range(16):
        k_local = r_w // 4
        i = r_w % 4
        k_global = k_local + 4
        W_full[k_global * 4 + i, offset:offset + 49] = W4_44[r_w, :]

    # Part 3: 4x4x1 (col 8 of B, row 8 of C)
    offset = 98
    U_full[:, offset:offset + 16] = U4_16
    for r_v in range(4):
        V_full[r_v * 9 + 8, offset:offset + 16] = V4_16[r_v, :]
    for r_w in range(4):
        W_full[8 * 4 + r_w, offset:offset + 16] = W4_16[r_w, :]

    return U_full, V_full, W_full


def _tensor_residual(U, V, W):
    """Compute tensor T[i,j,k] - 1 for all i,j,k. Shape (n,m,p)."""
    r = U.shape[1]
    U_r = U.reshape(N, M, r)
    V_r = V.reshape(M, P, r)
    W_r = W.reshape(P, N, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T - 1.0


def _is_valid(U, V, W, atol=1e-9):
    """Check exact decomposition."""
    r = U.shape[1]
    U_r = U.reshape(N, M, r)
    V_r = V.reshape(M, P, r)
    W_r = W.reshape(P, N, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=atol)


def _try_remove_column(U, V, W, col_idx):
    """
    Try to remove column col_idx by adjusting other columns.
    Returns (success, new_U, new_V, new_W)
    """
    r = U.shape[1]
    if r <= 1:
        return False, U, V, W

    # Residual if we simply zero out column col_idx
    U_mod = U.copy()
    V_mod = V.copy()
    W_mod = W.copy()
    U_mod[:, col_idx] = 0
    V_mod[:, col_idx] = 0
    W_mod[:, col_idx] = 0

    residual = _tensor_residual(U_mod, V_mod, W_mod)
    # residual[i,j,k] = - (U[:,col_idx] * V[:,col_idx] * W[:,col_idx]) contribution
    # We want to find delta_c for c != col_idx such that:
    # sum_{c != col_idx} delta_c * U[:,c] * V[:,c] * W[:,c] = -residual
    # This is a linear system in delta_c.
    # For each (i,j,k), the equation is:
    # sum_{c != col_idx} delta_c * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = -residual[i,j,k]

    other_cols = [c for c in range(r) if c != col_idx]
    num_other = len(other_cols)

    # Build A matrix: size (N*M*P, num_other)
    # A[(i,j,k), c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    A = np.zeros((N * M * P, num_other))
    b = np.zeros(N * M * P)

    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                b[idx] = -residual[i, j, k]
                for oc, c in enumerate(other_cols):
                    A[idx, oc] = U[i * M + j, c] * V[j * P + k, c] * W[k * N + i, c]
                idx += 1

    # Solve A * delta = b
    try:
        delta, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return False, U, V, W

    # Check if solution is exact
    A_delta = A @ delta
    if not np.allclose(A_delta, b, atol=1e-8):
        return False, U, V, W

    # Apply delta
    U_new = U_mod.copy()
    V_new = V_mod.copy()
    W_new = W_mod.copy()
    for oc, c in enumerate(other_cols):
        U_new[:, c] += delta[oc] * U[:, c]
        V_new[:, c] += delta[oc] * V[:, c]
        W_new[:, c] += delta[oc] * W[:, c]

    # Final verification
    if not _is_valid(U_new, V_new, W_new):
        return False, U, V, W

    # Remove the column
    keep = [c for c in range(r) if c != col_idx]
    return True, U_new[:, keep], V_new[:, keep], W_new[:, keep]


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Start with known 114-rank decomposition
    U, V, W = _build_block_decomp()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Iterative rank reduction
    # Try to remove columns one by one, always working from the current best
    # This is a greedy approach
    max_iterations = 1000
    iteration = 0

    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        current_rank = U.shape[1]
        if current_rank <= 1:
            break

        found_removal = False
        # Try each column
        for col in range(current_rank):
            if time.time() > deadline:
                break

            success, U_new, V_new, W_new = _try_remove_column(U, V, W, col)
            if success:
                U, V, W = U_new, V_new, W_new
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
                found_removal = True
                break  # Restart column search with new rank

        if not found_removal:
            # No more columns can be removed; try a different strategy
            # For now, we stop. A more advanced strategy might try
            # to swap columns or use different repair methods.
            break

    # Final verification
    if not _is_valid(best_U, best_V, best_W):
        # Fallback to schoolbook if something went wrong
        rank = N * M * P
        U_fb = np.zeros((N * M, rank))
        V_fb = np.zeros((M * P, rank))
        W_fb = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_fb[i * M + j, col] = 1.0
                    V_fb[j * P + k, col] = 1.0
                    W_fb[k * N + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
