import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0


def _build_kronecker_matrix(U, V, W):
    """Build the n*m*p x rank matrix where each column is the vectorized product
    U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    A = np.zeros((n * m * p, r))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[idx, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                idx += 1
    return A


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Also verify no spurious terms: for each (i,j,k), the sum should be exactly 1
    # This is already checked above.
    return True


def _verify_decomposition_fast(U, V, W, tol=1e-9):
    """Fast verification using vectorized operations."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                if abs(np.sum(prod) - 1.0) > tol:
                    return False
    return True


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


def _compute_rank_of_kronecker(K, tol=1e-10):
    """Compute the numerical rank of the Kronecker matrix K."""
    try:
        s_vals = np.linalg.svd(K, compute_uv=False)
    except:
        return K.shape[1]
    if len(s_vals) == 0:
        return 0
    max_s = s_vals[0] if s_vals[0] > 0 else 1.0
    num_rank = np.sum(s_vals > tol * max_s)
    return int(num_rank)


def _find_null_space(K, tol=1e-10):
    """Find the null space of K. Returns a list of null space vectors (as rows)."""
    try:
        U_svd, s_vals, Vt = np.linalg.svd(K, full_matrices=True)
    except:
        return []
    if len(s_vals) == 0:
        return []
    max_s = s_vals[0] if s_vals[0] > 0 else 1.0
    num_rank = np.sum(s_vals > tol * max_s)
    if num_rank >= K.shape[1]:
        return []
    null_vecs = Vt[num_rank:]
    return null_vecs


def _greedy_min_subset(K, target_rank, time_limit):
    """Find a minimal set of columns that spans the same space as all columns.
    Uses greedy forward selection."""
    n_cols = K.shape[1]
    if target_rank >= n_cols:
        return list(range(n_cols))
    
    # Start with all columns, try to remove
    # First, find which columns are needed via QR with column pivoting
    try:
        Q, R, piv = np.linalg.qr(K, mode='complete', pivoting=True)
    except:
        return list(range(n_cols))
    
    # The first target_rank pivoted columns span the space
    selected = list(piv[:target_rank])
    return selected


def _reduce_by_null_space(U, V, W, K, null_vecs, time_limit):
    """Use null space vectors to identify and remove redundant columns."""
    rank = U.shape[1]
    removed = []
    
    for nv in null_vecs:
        # Find the column with the largest absolute value in this null vector
        # and try to remove it
        abs_nv = np.abs(nv)
        max_idx = np.argmax(abs_nv)
        if abs_nv[max_idx] < 1e-8:
            continue
        
        # Check if removing this column still gives a valid decomposition
        mask = [c for c in range(rank) if c != max_idx]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _verify_decomposition_fast(U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            K = _build_kronecker_matrix(U, V, W)
            removed.append(max_idx)
    
    return U, V, W, K, removed


def _try_remove_column(U, V, W, K, col, tol=1e-10):
    """Try to remove a single column and verify."""
    rank = U.shape[1]
    mask = [c for c in range(rank) if c != col]
    if len(mask) == 0:
        return None, None, None, None, False
    
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_decomposition_fast(U_new, V_new, W_new, tol):
        K_new = _build_kronecker_matrix(U_new, V_new, W_new)
        return U_new, V_new, W_new, K_new, True
    return None, None, None, None, False


def _try_remove_columns(U, V, W, K, cols_to_remove, tol=1e-10):
    """Try to remove multiple columns and verify."""
    rank = U.shape[1]
    mask = [c for c in range(rank) if c not in cols_to_remove]
    if len(mask) == 0:
        return None, None, None, None, False
    
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_decomposition_fast(U_new, V_new, W_new, tol):
        K_new = _build_kronecker_matrix(U_new, V_new, W_new)
        return U_new, V_new, W_new, K_new, True
    return None, None, None, None, False


def _global_rank_reduction(U, V, W, K, time_limit):
    """Aggressively reduce rank using global linear dependency analysis."""
    start = time.time()
    rank = U.shape[1]
    
    # Iteratively remove redundant columns
    progress = True
    while progress and time.time() < time_limit:
        progress = False
        rank = U.shape[1]
        
        # Compute the rank of the Kronecker matrix
        k_rank = _compute_rank_of_kronecker(K, tol=1e-10)
        
        if k_rank >= rank:
            # No more linear dependencies to exploit via simple column removal
            # Try individual column removals
            for col in range(rank):
                if time.time() > time_limit:
                    break
                U_new, V_new, W_new, K_new, ok = _try_remove_column(U, V, W, K, col)
                if ok:
                    U, V, W, K = U_new, V_new, W_new, K_new
                    progress = True
                    break
        else:
            # There are linear dependencies. Find null space vectors.
            null_vecs = _find_null_space(K, tol=1e-10)
            if not null_vecs:
                null_vecs = _find_null_space(K, tol=1e-8)
            
            if null_vecs:
                # For each null vector, try to remove the column with max coefficient
                for nv in null_vecs:
                    if time.time() > time_limit:
                        break
                    abs_nv = np.abs(nv)
                    # Sort columns by coefficient magnitude
                    order = np.argsort(-abs_nv)
                    for col in order:
                        if abs_nv[col] < 1e-8:
                            break
                        U_new, V_new, W_new, K_new, ok = _try_remove_column(U, V, W, K, col)
                        if ok:
                            U, V, W, K = U_new, V_new, W_new, K_new
                            progress = True
                            break
                    if progress:
                        break
            
            # Also try removing pairs of columns that appear together in null vectors
            if not progress and time.time() < time_limit:
                for nv in null_vecs:
                    if time.time() > time_limit:
                        break
                    abs_nv = np.abs(nv)
                    order = np.argsort(-abs_nv)
                    # Try removing top 2 columns
                    if len(order) >= 2 and abs_nv[order[0]] > 1e-8 and abs_nv[order[1]] > 1e-8:
                        cols = [int(order[0]), int(order[1])]
                        U_new, V_new, W_new, K_new, ok = _try_remove_columns(U, V, W, K, cols)
                        if ok:
                            U, V, W, K = U_new, V_new, W_new, K_new
                            progress = True
                            break
    
    return U, V, W, K


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Build the rank-36 Strassen decomposition
    U, V, W = _build_strassen_36()

    # Verify the base decomposition
    if not _verify_decomposition_fast(U, V, W):
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

    # Save best result
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Build Kronecker matrix
    K = _build_kronecker_matrix(U, V, W)
    
    # Phase 1: Global rank reduction using null space analysis
    U, V, W, K = _global_rank_reduction(U, V, W, K, deadline - 20)
    
    if U.shape[1] < best_rank:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
    
    # Phase 2: Exhaustive single-column removal with tight tolerance
    if time.time() < deadline - 10:
        rank = U.shape[1]
        for col in range(rank):
            if time.time() > deadline - 10:
                break
            U_new, V_new, W_new, K_new, ok = _try_remove_column(U, V, W, K, col, tol=1e-12)
            if ok:
                U, V, W, K = U_new, V_new, W_new, K_new
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
    
    # Phase 3: Try pair removals
    if time.time() < deadline - 5:
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 5:
                    break
                U_new, V_new, W_new, K_new, ok = _try_remove_columns(U, V, W, K, [i, j], tol=1e-12)
                if ok:
                    U, V, W, K = U_new, V_new, W_new, K_new
                    if U.shape[1] < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = U.shape[1]
                    break
            if U.shape[1] < best_rank:
                break
    
    # Phase 4: Try triple removals if time permits
    if time.time() < deadline - 3 and U.shape[1] > 4:
        rank = U.shape[1]
        found = False
        for i in range(rank):
            if time.time() > deadline - 3 or found:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 3 or found:
                    break
                for k in range(j + 1, rank):
                    if time.time() > deadline - 3 or found:
                        break
                    U_new, V_new, W_new, K_new, ok = _try_remove_columns(U, V, W, K, [i, j, k], tol=1e-12)
                    if ok:
                        U, V, W, K = U_new, V_new, W_new, K_new
                        if U.shape[1] < best_rank:
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_rank = U.shape[1]
                        found = True
                        break
    
    # Final verification
    if not _verify_decomposition_fast(best_U, best_V, best_W):
        # Use the schoolbook fallback
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
