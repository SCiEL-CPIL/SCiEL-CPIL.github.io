"""
Strategy: Data-driven rank minimization via iterative column pruning and merging.

Instead of a fixed block-Strassen structure (rank 88), we start from a valid
decomposition and iteratively:
1. Try zeroing out columns that contribute nothing (redundant terms).
2. Try merging pairs of columns whose outer products are linearly dependent.
3. Try small perturbations to find equivalent lower-rank forms.

This is a fundamentally different optimization approach: local search over the
space of valid decompositions, rather than a fixed algebraic construction.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _build_schoolbook():
    """Build the schoolbook (rank 96) decomposition."""
    n, m, p = DIMENSIONS
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


def _build_strassen_blocks():
    """Build the block-Strassen decomposition (rank 88)."""
    n, m, p = DIMENSIONS
    # j into 2 blocks of 2, k into 4 blocks of 2
    # Each block: Strassen (7) for i={0,1} + schoolbook (4) for i={2}
    rank_per_block = 11
    total_blocks = 2 * 4
    rank = rank_per_block * total_blocks

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 terms (a_coeffs for A, b_coeffs for B, c_coeffs for C)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]),
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, -1, 0]),
        ([1, 0, 0, 0], [1, 0, -1, 0], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen for i={0,1}, j={j0,j0+1}, k={k0,k0+1}
            for t in range(7):
                a_c, b_c, c_c = strassen_terms[t]
                for di in range(2):
                    for dj in range(2):
                        U[(di) * m + (j0 + dj), col] = a_c[di * 2 + dj]
                for dj in range(2):
                    for dk in range(2):
                        V[(j0 + dj) * p + (k0 + dk), col] = b_c[dj * 2 + dk]
                for di in range(2):
                    for dk in range(2):
                        W[(k0 + dk) * n + di, col] = c_c[di * 2 + dk]
                col += 1
            # Schoolbook for i={2}, j={j0,j0+1}, k={k0,k0+1}
            for dj in range(2):
                for dk in range(2):
                    U[2 * m + (j0 + dj), col] = 1
                    V[(j0 + dj) * p + (k0 + dk), col] = 1
                    W[(k0 + dk) * n + 2, col] = 1
                    col += 1

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = U.shape[1]
    # Build the tensor check via matrix operations
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape for batch computation
    U_r = U.T.reshape(rank, n, m)  # rank x n x m
    V_r = V.T.reshape(rank, m, p)  # rank x m x p
    W_r = W.T.reshape(rank, p, n)  # rank x p x n

    # For each (i,j,k): sum_c U_r[c,i,j] * V_r[c,j,k] * W_r[c,k,i]
    # This is a 4-way contraction. Do it in chunks for memory.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U_r[:, i, j], V_r[:, j, k]), W_r[:, k, i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_remove_column(U, V, W, n, m, p, col_idx):
    """Try removing column col_idx and check if the remaining decomposition is still valid.
    Returns (success, new_U, new_V, new_W)."""
    if col_idx < 0 or col_idx >= U.shape[1]:
        return False, None, None, None

    U_new = np.delete(U, col_idx, axis=1)
    V_new = np.delete(V, col_idx, axis=1)
    W_new = np.delete(W, col_idx, axis=1)

    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    return False, None, None, None


def _try_merge_columns(U, V, W, n, m, p, c1, c2):
    """Try merging columns c1 and c2 by expressing one as a combination of the other.
    If the two columns are proportional, we can remove one.
    Returns (success, new_U, new_V, new_W)."""
    if c1 == c2 or c1 < 0 or c2 < 0 or c1 >= U.shape[1] or c2 >= U.shape[1]:
        return False, None, None, None

    # Check if column c1 is a scalar multiple of column c2
    # If so, we can absorb it
    u1, u2 = U[:, c1], U[:, c2]
    v1, v2 = V[:, c1], V[:, c2]
    w1, w2 = W[:, c1], W[:, c2]

    # Try: can we write the contribution of c1 as a scaled version of c2?
    # This requires u1 = alpha*u2, v1 = alpha*v2, w1 = beta*w2 with alpha*beta = 1
    # More generally, we need the rank-1 tensor from c1 to be a multiple of c2's.

    # Simple check: if all three vectors are proportional
    # Find alpha from U
    nonzero_u2 = np.abs(u2) > 1e-12
    if not np.any(nonzero_u2):
        return False, None, None, None

    alphas = u1[nonzero_u2] / u2[nonzero_u2]
    # Check consistency
    if np.std(alphas) > 1e-9:
        return False, None, None, None
    alpha = alphas[0]

    # Check V
    if np.any(np.abs(v1 - alpha * v2) > 1e-9 * max(1, np.max(np.abs(v1)), np.max(np.abs(v2)))):
        return False, None, None, None

    # Check W: need w1 = beta * w2 where alpha * beta = 1, so beta = 1/alpha
    if abs(alpha) < 1e-12:
        return False, None, None, None
    beta = 1.0 / alpha
    if np.any(np.abs(w1 - beta * w2) > 1e-9 * max(1, np.max(np.abs(w1)), np.max(np.abs(w2)))):
        return False, None, None, None

    # Columns are proportional - remove c1
    U_new = np.delete(U, c1, axis=1)
    V_new = np.delete(V, c1, axis=1)
    W_new = np.delete(W, c1, axis=1)
    return True, U_new, V_new, W_new


def _random_perturbation(U, V, W, n, m, p, num_trials=50):
    """Try random small perturbations to find equivalent lower-rank forms.
    We perturb and then try to re-orthogonalize or simplify."""
    best = (U, V, W)
    best_rank = U.shape[1]

    for trial in range(num_trials):
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()

        # Randomly zero out a small number of entries
        num_zeros = np.random.randint(1, 5)
        for _ in range(num_zeros):
            mat = np.random.choice([U_p, V_p, W_p])
            r, c = np.random.randint(mat.shape[0]), np.random.randint(mat.shape[1])
            mat[r, c] = 0.0

        if _verify_fast(U_p, V_p, W_p, n, m, p):
            # Try to remove more columns
            U_t, V_t, W_t = U_p, V_p, W_p
            for c in range(U_t.shape[1] - 1, -1, -1):
                ok, U_t, V_t, W_t = _try_remove_column(U_t, V_t, W_t, n, m, p, c)
                if ok:
                    continue
            if U_t.shape[1] < best_rank:
                best = (U_t, V_t, W_t)
                best_rank = U_t.shape[1]

    return best


def _greedy_column_elimination(U, V, W, n, m, p, max_iterations=200):
    """Iteratively try to remove columns until no more can be removed."""
    for iteration in range(max_iterations):
        removed_any = False
        for c in range(U.shape[1] - 1, -1, -1):
            ok, U_new, V_new, W_new = _try_remove_column(U, V, W, n, m, p, c)
            if ok:
                U, V, W = U_new, V_new, W_new
                removed_any = True
                break
        if not removed_any:
            break
    return U, V, W


def _try_column_combination(U, V, W, n, m, p, c1, c2, max_iter=50):
    """Try to express column c1 as a linear combination of c2 and other columns,
    then eliminate c1. This is a more general merging strategy."""
    rank = U.shape[1]
    if c1 == c2 or c1 < 0 or c2 < 0 or c1 >= rank or c2 >= rank:
        return False, None, None, None

    # We want to find coefficients such that:
    # U[:,c1] = sum_{c != c1} alpha_c * U[:,c]
    # V[:,c1] = sum_{c != c1} alpha_c * V[:,c]  (same alpha!)
    # W[:,c1] = sum_{c != c1} alpha_c * W[:,c]  (same alpha!)

    # This is a linear system. Set up:
    # For U: U[:,c1] = U[:, others] @ alpha
    # For V: V[:,c1] = V[:, others] @ alpha
    # For W: W[:,c1] = W[:, others] @ alpha

    others = [c for c in range(rank) if c != c1]
    U_others = U[:, others]
    V_others = V[:, others]
    W_others = W[:, others]

    # Solve least squares for all three simultaneously
    # Stack: [U_others; V_others; W_others] @ alpha = [U[:,c1]; V[:,c1]; W[:,c1]]
    A_stack = np.vstack([U_others.T, V_others.T, W_others.T])  # (3*rank-3) x (rank-1)
    b_stack = np.concatenate([U[:, c1], V[:, c1], W[:, c1]])  # 3*(rank-1)

    try:
        alpha, residuals, rank_s, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
    except:
        return False, None, None, None

    # Check residual
    if residuals and residuals[0] > 1e-6:
        return False, None, None, None

    # Verify
    U_pred = U_others @ alpha
    V_pred = V_others @ alpha
    W_pred = W_others @ alpha

    if (np.max(np.abs(U[:, c1] - U_pred)) > 1e-8 or
            np.max(np.abs(V[:, c1] - V_pred)) > 1e-8 or
            np.max(np.abs(W[:, c1] - W_pred)) > 1e-8):
        return False, None, None, None

    # Remove column c1
    U_new = np.delete(U, c1, axis=1)
    V_new = np.delete(V, c1, axis=1)
    W_new = np.delete(W, c1, axis=1)
    return True, U_new, V_new, W_new


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Start with the Strassen block decomposition (rank 88)
    U, V, W = _build_strassen_blocks()
    current_rank = U.shape[1]

    # Verify initial
    if not _verify_fast(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _build_schoolbook()
        current_rank = U.shape[1]

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    # Phase 1: Greedy column elimination
    U, V, W = _greedy_column_elimination(U, V, W, n, m, p)
    current_rank = U.shape[1]
    if current_rank < best_rank:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank

    # Check time
    if time.time() > deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Phase 2: Try merging proportional columns
    for c1 in range(best_U.shape[1]):
        if time.time() > deadline - 30:
            break
        for c2 in range(c1 + 1, best_U.shape[1]):
            if time.time() > deadline - 30:
                break
            ok, U_new, V_new, W_new = _try_merge_columns(
                best_U, best_V, best_W, n, m, p, c1, c2)
            if ok:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = best_U.shape[1]
                c1 = -1  # Restart since indices changed
                break

    # Check time
    if time.time() > deadline - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Phase 3: Try linear combination elimination (more general)
    for c1 in range(best_U.shape[1]):
        if time.time() > deadline - 20:
            break
        ok, U_new, V_new, W_new = _try_column_combination(
            best_U, best_V, best_W, n, m, p, c1, -1)
        if ok:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = best_U.shape[1]
            c1 = -1  # Restart

    # Check time
    if time.time() > deadline - 15:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Phase 4: Random perturbations
    if time.time() < deadline - 10:
        U_r, V_r, W_r = _random_perturbation(
            best_U, best_V, best_W, n, m, p,
            num_trials=min(20, int((deadline - time.time()) / 0.5)))
        if U_r.shape[1] < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = U_r.shape[1]

    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
