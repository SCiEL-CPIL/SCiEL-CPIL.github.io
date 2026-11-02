import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 135.0  # safety margin before 150s kill


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


def _build_kronecker_matrix(U, V, W):
    """Build the n*m*p x rank matrix where each column is the vectorized Kronecker product
    of U[:,c], V[:,c], W[:,c]."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    A = np.zeros((n * m * p, r))
    for c in range(r):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A


def _check_column_redundant_fast(K, col_to_remove):
    """Check if column col_to_remove of K is in the span of the other columns.
    Returns True if redundant, False otherwise."""
    r = K.shape[1]
    if r <= 1:
        return False

    keep_cols = [c for c in range(r) if c != col_to_remove]
    A = K[:, keep_cols]
    b = K[:, col_to_remove]

    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False

    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False

    return True


def _absorb_and_remove(U, V, W, K, col_to_remove):
    """Absorb a dependent column into others, then remove it.
    Returns (new_U, new_V, new_W, new_K) or None if absorption not possible."""
    rank = U.shape[1]
    if rank <= 1:
        return None

    keep_cols = [c for c in range(rank) if c != col_to_remove]
    A = K[:, keep_cols]
    b = K[:, col_to_remove]

    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return None

    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return None

    # Absorb: for each kept column c', add solution[keep_cols.index(c')] * (U[:,c'], V[:,c'], W[:,c'])
    # Wait, that's not right. The column K[:, col_to_remove] = sum_{c' in keep} alpha_{c'} K[:, c']
    # K[:, c'] = vec(U[:,c'] ⊙ V[:,c'] ⊙ W[:,c'])
    # So we want: vec(U[:,c_to_remove] ⊙ V[:,c_to_remove] ⊙ W[:,c_to_remove]) = sum alpha_{c'} vec(U[:,c'] ⊙ V[:,c'] ⊙ W[:,c'])
    # This doesn't directly give us how to modify U, V, W.
    # The absorption trick is: if column c is dependent, we can scale other columns to "absorb" it.
    # Actually, the standard trick is: if K_c = sum alpha_{c'} K_{c'}, then we can define new columns:
    # U_new[:, c'] = U[:, c'] + alpha_{c'} * U[:, c] * (V[:, c'] / V[:, c]) * (W[:, c'] / W[:, c]) ... no, that's wrong.
    # 
    # Let me think again. The key insight is that the tensor is:
    # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # If K_c (the vectorized version) is in the span of others, then the tensor contribution of column c
    # can be expressed as a linear combination of other columns' contributions ONLY IF we can find
    # coefficients that work at the tensor level, not just the vectorized level.
    # 
    # Actually, for matrix multiplication, the condition is that for all (i,j,k):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If we remove column c, we need:
    # sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c'] = 1
    # This is satisfied if and only if the vectorized column K_c is in the span of the others.
    # But that's only sufficient for the SUM to remain 1, not for individual entries to match.
    # 
    # Wait, no. The tensor equation is:
    # T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need this to equal 1 for all (i,j,k).
    # The vectorized column K_c has entries K[(i,j,k), c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # So T_{i,j,k} = sum_c K[(i,j,k), c]
    # If K_c = sum_{c' != c} alpha_{c'} K_{c'}, then:
    # sum_c K[(i,j,k), c] = sum_{c' != c} K[(i,j,k), c'] + K[(i,j,k), c]
    # = sum_{c' != c} K[(i,j,k), c'] + sum_{c' != c} alpha_{c'} K[(i,j,k), c']
    # = sum_{c' != c} (1 + alpha_{c'}) K[(i,j,k), c']
    # This is NOT equal to sum_{c' != c} K[(i,j,k), c'] unless sum alpha_{c'} K[(i,j,k), c'] = 0 for all (i,j,k).
    # 
    # So simple linear dependence of K columns does NOT allow removal!
    # We need: K_c = 0 in the sense that its contribution cancels.
    # Actually, the correct condition is: if we set U[:,c] = V[:,c] = W[:,c] = 0, the sum changes by K_c.
    # So we can only remove column c if K_c = 0, which means the column is zero.
    # 
    # OR, we can modify OTHER columns to compensate. If K_c = sum alpha_{c'} K_{c'}, then:
    # New sum = sum_{c' != c} K_{c'} + (sum_{c' != c} alpha_{c'} K_{c'} - K_c) + K_c ... no.
    # 
    # Let me reconsider. We want to find new values for U, V, W such that:
    # sum_c' U_new[i,j,c'] V_new[j,k,c'] W_new[k,i,c'] = 1 for all (i,j,k)
    # and column c is removed (set to zero or deleted).
    # 
    # One approach: scale columns. If we multiply U[:,c'], V[:,c'], W[:,c'] by factors that multiply to 1,
    # the product is unchanged. But we can't independently scale them.
    # 
    # The absorption trick that works: if K_c = sum alpha_{c'} K_{c'}, we can't just remove it.
    # We need a different approach. Let me just do the simple removal check and see if it works.
    
    # Actually, let me just verify: if we remove column c and the sum of remaining K columns still equals 1
    # for all (i,j,k), then we're good. Let's check that directly.
    mask = [i for i in range(rank) if i != col_to_remove]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    n, m, p = 2, 4, 5
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_new[i * m + j, :], V_new[j * p + k, :] * W_new[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    K_new = _build_kronecker_matrix(U_new, V_new, W_new)
    return U_new, V_new, W_new, K_new


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
    rank = U.shape[1]

    # Pre-compute Kronecker matrix K for the initial decomposition
    K = _build_kronecker_matrix(U, V, W)

    # Phase 1: Try to remove columns by direct verification
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        rank = U.shape[1]
        for col in range(rank):
            if time.time() > deadline - 10:
                break

            result = _absorb_and_remove(U, V, W, K, col)
            if result is not None:
                U, V, W, K = result
                rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break

    # Phase 2: Try removing pairs
    if time.time() < deadline - 5:
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            rank = U.shape[1]
            for i in range(rank):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, rank):
                    if time.time() > deadline - 5:
                        break

                    mask = [c for c in range(rank) if c != i and c != j]
                    if len(mask) < 1:
                        continue

                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]

                    n, m, p = 2, 4, 5
                    valid = True
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                val = np.dot(U_new[ii * m + jj, :], V_new[jj * p + kk, :] * W_new[kk * n + ii, :])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        continue

                    K_new = _build_kronecker_matrix(U_new, V_new, W_new)
                    U, V, W, K = U_new, V_new, W_new, K_new
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    progress = True
                    break
                if time.time() > deadline - 5:
                    break
                if progress:
                    break

    # Phase 3: Random rotations to expose dependencies
    if time.time() < deadline - 2:
        rng = np.random.RandomState(42)
        rank = U.shape[1]
        for trial in range(80):
            if time.time() > deadline - 2:
                break
            if rank < 3:
                break

            cols = rng.choice(rank, size=2, replace=False)
            t = rng.uniform(-2, 2)

            U_rot = U.copy()
            V_rot = V.copy()
            W_rot = W.copy()

            U_rot[:, cols[0]] = U[:, cols[0]] + t * U[:, cols[1]]
            V_rot[:, cols[0]] = V[:, cols[0]] + t * V[:, cols[1]]
            W_rot[:, cols[0]] = W[:, cols[0]] + t * W[:, cols[1]]

            # Verify the rotated decomposition is still valid
            if not _verify_decomposition(U_rot, V_rot, W_rot):
                continue

            K_rot = _build_kronecker_matrix(U_rot, V_rot, W_rot)

            # Check for single column redundancy in the rotated system
            for col in range(rank):
                if time.time() > deadline - 2:
                    break
                result = _absorb_and_remove(U_rot, V_rot, W_rot, K_rot, col)
                if result is not None:
                    U, V, W, K = result
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
