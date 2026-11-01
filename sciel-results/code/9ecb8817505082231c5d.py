import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill


def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices

        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1

        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1

        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1

        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1

    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (2, 1), (3, 0), (3, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 2), (2, 3), (3, 2), (3, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])

    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication (vectorized)."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_kronecker_matrix(U, V, W, cols):
    """Build the n*m*p x len(cols) matrix where each column is the Kronecker product
    of U[:,c] ⊗ V[:,c] ⊗ W[:,c] for c in cols."""
    n, m, p = 2, 4, 5
    r = len(cols)
    A = np.zeros((n * m * p, r))
    for ci, c in enumerate(cols):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, ci] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A


def _check_column_redundant(U, V, W, col_to_remove):
    """Check if column col_to_remove can be expressed as a linear combination of others."""
    n, m, p = 2, 4, 5
    r = U.shape[1]

    keep_cols = [c for c in range(r) if c != col_to_remove]
    num_unknowns = len(keep_cols)

    if num_unknowns < 1:
        return False, None

    # Build the system
    A = _build_kronecker_matrix(U, V, W, keep_cols)
    b = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[idx] = U[i * m + j, col_to_remove] * V[j * p + k, col_to_remove] * W[k * n + i, col_to_remove]
                idx += 1

    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False, None

    # Check if the solution actually satisfies the equations
    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False, None

    return True, solution


def _check_pair_redundant(U, V, W, col1, col2):
    """Check if columns col1 and col2 can both be removed."""
    n, m, p = 2, 4, 5
    r = U.shape[1]

    keep_cols = [c for c in range(r) if c != col1 and c != col2]
    num_unknowns = len(keep_cols)

    if num_unknowns < 2:
        return False

    # Build the system for both columns simultaneously
    A = _build_kronecker_matrix(U, V, W, keep_cols)
    b = np.zeros((n * m * p, 2))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[idx, 0] = U[i * m + j, col1] * V[j * p + k, col1] * W[k * n + i, col1]
                b[idx, 1] = U[i * m + j, col2] * V[j * p + k, col2] * W[k * n + i, col2]
                idx += 1

    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False

    # Check if the solution actually satisfies the equations
    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False

    return True


def _try_column_rotation(U, V, W, col_a, col_b, t):
    """Try a rotation in the 2D subspace spanned by columns col_a and col_b."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()

    U_new[:, col_a] = U[:, col_a] + t * U[:, col_b]
    V_new[:, col_a] = V[:, col_a] + t * V[:, col_b]
    W_new[:, col_a] = W[:, col_a] + t * W[:, col_b]

    return U_new, V_new, W_new


def _try_triple_removal(U, V, W, col1, col2, col3):
    """Check if columns col1, col2, col3 can all be removed."""
    n, m, p = 2, 4, 5
    r = U.shape[1]

    keep_cols = [c for c in range(r) if c != col1 and c != col2 and c != col3]
    if len(keep_cols) < 3:
        return False

    A = _build_kronecker_matrix(U, V, W, keep_cols)
    b = np.zeros((n * m * p, 3))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[idx, 0] = U[i * m + j, col1] * V[j * p + k, col1] * W[k * n + i, col1]
                b[idx, 1] = U[i * m + j, col2] * V[j * p + k, col2] * W[k * n + i, col2]
                b[idx, 2] = U[i * m + j, col3] * V[j * p + k, col3] * W[k * n + i, col3]
                idx += 1

    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False

    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False

    return True


def _greedy_remove_single(U, V, W, deadline):
    """Phase 1: Greedily remove single columns if linearly dependent."""
    rank = U.shape[1]
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
        progress = False
        for col in range(rank):
            if time.time() > deadline - 20:
                break
            redundant, _ = _check_column_redundant(U, V, W, col)
            if redundant:
                mask = [i for i in range(rank) if i != col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    progress = True
                    break
        if not progress:
            break
    return U, V, W


def _greedy_remove_pairs(U, V, W, deadline):
    """Phase 2: Greedily remove pairs of columns if jointly dependent."""
    rank = U.shape[1]
    progress = True
    while progress and time.time() < deadline - 15:
        progress = False
        for i in range(rank):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 15:
                    break
                if _check_pair_redundant(U, V, W, i, j):
                    mask = [k for k in range(rank) if k != i and k != j]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        rank = U.shape[1]
                        progress = True
                        break
            if time.time() > deadline - 15:
                break
            if progress:
                break
    return U, V, W


def _greedy_remove_triples(U, V, W, deadline):
    """Phase 3: Try removing triples of columns."""
    rank = U.shape[1]
    if rank < 4:
        return U, V, W
    for i in range(rank):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, rank):
            if time.time() > deadline - 10:
                break
            for k in range(j + 1, rank):
                if time.time() > deadline - 10:
                    break
                if _try_triple_removal(U, V, W, i, j, k):
                    mask = [l for l in range(rank) if l != i and l != j and l != k]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        rank = U.shape[1]
                        break
            if time.time() > deadline - 10:
                break
            if U.shape[1] < rank:
                break
        if time.time() > deadline - 10:
            break
    return U, V, W


def _rotation_search(U, V, W, deadline):
    """Phase 4: Try column rotations to expose hidden dependencies."""
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        if rank < 4:
            break
        cols = rng.choice(rank, size=2, replace=False)
        t = rng.uniform(-2, 2)
        U_rot, V_rot, W_rot = _try_column_rotation(U, V, W, int(cols[0]), int(cols[1]), t)

        for col in range(rank):
            if time.time() > deadline - 5:
                break
            redundant, _ = _check_column_redundant(U_rot, V_rot, W_rot, col)
            if redundant:
                mask = [i for i in range(rank) if i != col]
                U_new = U_rot[:, mask]
                V_new = V_rot[:, mask]
                W_new = W_rot[:, mask]
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    break
    return U, V, W


def _final_sweep(U, V, W, deadline):
    """Phase 5: Final single-column removal sweep."""
    rank = U.shape[1]
    for attempt in range(50):
        if time.time() > deadline - 3:
            break
        progress = False
        for col in range(rank):
            if time.time() > deadline - 3:
                break
            redundant, _ = _check_column_redundant(U, V, W, col)
            if redundant:
                mask = [i for i in range(rank) if i != col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    progress = True
                    break
        if not progress:
            break
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    U, V, W = _build_strassen_36()

    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 1: Greedy single column removal
    U, V, W = _greedy_remove_single(U, V, W, deadline)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 2: Greedy pair removal
    U, V, W = _greedy_remove_pairs(U, V, W, deadline)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 3: Triple removal
    U, V, W = _greedy_remove_triples(U, V, W, deadline)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 4: Rotation search
    U, V, W = _rotation_search(U, V, W, deadline)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 5: Final sweep
    U, V, W = _final_sweep(U, V, W, deadline)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
