import numpy as np
import time

DIMENSIONS = [3, 3, 6]


def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.

    U: (4, 7) where U[i*2+j, r] for A[i,j]
    V: (4, 7) where V[i*2+j, r] for B[i,j]
    W: (4, 7) where W[i*2+j, r] for C[i,j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1

    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1

    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1

    return U, V, W


def _build_block_decomposition(n, m, p):
    """Block decomposition with Strassen for 2x2 blocks.

    For 3x3 * 3x6:
      Split B into three 3x2 blocks (columns 0-1, 2-3, 4-5).
      For each 3x2 block of B:
        - Top 2 rows of C: A_top(2x3) * Bj_top(2x2) [Strassen rank 7]
                           + A_top_right(2x1) * Bj_bottom(1x2) [4 rank-1]
        - Bottom 1 row of C: A_bot(1x3) * Bj_top(2x2) [4 rank-1]
                           + A_bot_right(1x1) * Bj_bottom(1x2) [2 rank-1]
      Total per block: 7 + 4 + 4 + 2 = 17
      Total rank = 3 * 17 = 51
    """
    n, m, p = 3, 3, 6

    U_cols = []
    V_cols = []
    W_cols = []

    for block in range(3):
        b_col_start = 2 * block

        U_s, V_s, W_s = _strassen_2x2()

        for r in range(7):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)

            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * m + aj] = U_s[a_flat, r]

            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * p + global_bk] = V_s[b_flat, r]

            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * n + ci] = W_s[c_flat, r]

            U_cols.append(U_term)
            V_cols.append(V_term)
            W_cols.append(W_term)

        for i in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                global_k = b_col_start + k
                U_term[i * m + 2] = 1.0
                V_term[2 * p + global_k] = 1.0
                W_term[global_k * n + i] = 1.0
                U_cols.append(U_term)
                V_cols.append(V_term)
                W_cols.append(W_term)

        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                global_k = b_col_start + k
                U_term[2 * m + jj] = 1.0
                V_term[jj * p + global_k] = 1.0
                W_term[global_k * n + 2] = 1.0
                U_cols.append(U_term)
                V_cols.append(V_term)
                W_cols.append(W_term)

        for k in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            global_k = b_col_start + k
            U_term[2 * m + 2] = 1.0
            V_term[2 * p + global_k] = 1.0
            W_term[global_k * n + 2] = 1.0
            U_cols.append(U_term)
            V_cols.append(V_term)
            W_cols.append(W_term)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    rank = U.shape[1]
    return rank, U, V, W


def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of the tensor decomposition."""
    R = U.shape[1]
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n

    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))

    for a, b, c in nonzero:
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(s - 1.0) > tol:
            return False

    nonzero_set = set(nonzero)
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                if (a, b, c) not in nonzero_set:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > tol:
                        return False

    return True


def _verify_sampled(U, V, W, n, m, p, tol=1e-9, seed=42, max_checks=5000):
    """Sampled verification for speed during search."""
    R = U.shape[1]
    rng = np.random.RandomState(seed)
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(s - 1.0) > tol:
                    return False

    checked = 0
    while checked < max_checks:
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        a = i * m + j

        j2 = rng.randint(0, m)
        k2 = rng.randint(0, p)
        b = j2 * p + k2

        i2 = rng.randint(0, n)
        k3 = rng.randint(0, p)
        c = k3 * n + i2

        if j2 == j and i2 == i and k3 == k:
            continue

        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(s) > tol:
            return False
        checked += 1

    return True


def _compute_loss(U, V, W, n, m, p):
    """Compute the loss for the tensor decomposition."""
    R = U.shape[1]
    total = 0.0

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                total += (s - 1.0) ** 2

    rng = np.random.RandomState(0)
    for _ in range(300):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        a = i * m + j
        j2 = rng.randint(0, m)
        k2 = rng.randint(0, p)
        b = j2 * p + k2
        i2 = rng.randint(0, n)
        k3 = rng.randint(0, p)
        c = k3 * n + i2
        if j2 == j and i2 == i and k3 == k:
            continue
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        total += s ** 2

    return total


def _als_refine(U, V, W, n, m, p, target_rank, deadline, seed=0, max_iters=300):
    """ALS-style refinement with sparsity thresholding.

    Initialize with a random mixing of the schoolbook decomposition, then
    iteratively refine each factor matrix via least-squares while enforcing
    a hard sparsity threshold to keep the structure low-rank.
    """
    rng = np.random.RandomState(seed)
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n

    U0, V0, W0 = _build_schoolbook(n, m, p)
    R0 = U0.shape[1]

    # Build the tensor T[a, b, c] = 1 for nonzero entries
    # We'll work with the "flattened" formulation:
    # For each factor r, the contribution is U[:, r] * V[:, r] * W[:, r]
    # We want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]

    # Initialize via random mixing of schoolbook columns
    M = rng.randn(R0, target_rank)
    cur_U = (U0 @ M).copy()
    cur_V = (V0 @ M).copy()
    cur_W = (W0 @ M).copy()

    # Precompute the list of nonzero (a, b, c) and sampled zero entries
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))

    # Sample zero entries for the loss
    zero_samples = []
    checked = 0
    while checked < 500:
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        a = i * m + j
        j2 = rng.randint(0, m)
        k2 = rng.randint(0, p)
        b = j2 * p + k2
        i2 = rng.randint(0, n)
        k3 = rng.randint(0, p)
        c = k3 * n + i2
        if j2 == j and i2 == i and k3 == k:
            continue
        zero_samples.append((a, b, c))
        checked += 1

    all_entries = nonzero + zero_samples
    target_vals = [1.0] * len(nonzero) + [0.0] * len(zero_samples)

    best_loss = _compute_loss(cur_U, cur_V, cur_W, n, m, p)
    best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()

    sparsity_thresh = 1e-4

    for it in range(max_iters):
        if time.time() > deadline - 5:
            break

        # ALS: update U
        # For each column r, U[:, r] is updated by solving:
        # minimize sum_{(a,b,c)} (U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
        # This is a linear system in U[:, r]
        # grad wrt U[a, r] = sum_{(b,c)} V[b,r]*W[c,r]*(pred - T)
        # Hessian wrt U[a, r], U[a', r] = sum_{(b,c)} V[b,r]*W[c,r]^2 * delta(a,a')
        # So U[:, r] = (V[r,:] * W[r,:]) weighted least squares

        # Efficient ALS: for each r, the effective "B" matrix is V[:, r] * W[:, r]
        # We solve: minimize || (V[:,r] * W[:,r]) outer U[:,r] - T_vec ||^2
        # which is a linear system

        # Actually, the gradient for U[a, r] is:
        #   sum over all entries (a', b, c) where a' = a: V[b, r] * W[c, r] * (pred - T)
        # The Hessian for U[a, r] is:
        #   sum over all entries (a, b, c): V[b, r] * W[c, r]^2

        # For speed, let's do a simpler coordinate/gradient approach
        # with a small step

        # Compute current predictions
        pred_nonzero = []
        for a, b, c in nonzero:
            pred_nonzero.append(np.sum(cur_U[a, :] * cur_V[b, :] * cur_W[c, :]))
        pred_nonzero = np.array(pred_nonzero)

        pred_zero = []
        for a, b, c in zero_samples:
            pred_zero.append(np.sum(cur_U[a, :] * cur_V[b, :] * cur_W[c, :]))
        pred_zero = np.array(pred_zero)

        cur_loss = np.sum((pred_nonzero - 1.0) ** 2) + np.sum(pred_zero ** 2)

        # Gradient descent with adaptive step
        lr = 0.005
        improved = False

        # Update U
        for a in range(a_dim):
            if time.time() > deadline - 5:
                break
            grad = np.zeros(target_rank)
            # nonzero entries with this a
            for idx, (aa, b, c) in enumerate(nonzero):
                if aa == a:
                    grad += cur_V[b, :] * cur_W[c, :] * (pred_nonzero[idx] - 1.0)
            for idx, (aa, b, c) in enumerate(zero_samples):
                if aa == a:
                    grad += cur_V[b, :] * cur_W[c, :] * pred_zero[idx]
            cur_U[a, :] -= lr * grad

        # Update V
        for b in range(b_dim):
            if time.time() > deadline - 5:
                break
            grad = np.zeros(target_rank)
            for idx, (a, bb, c) in enumerate(nonzero):
                if bb == b:
                    grad += cur_U[a, :] * cur_W[c, :] * (pred_nonzero[idx] - 1.0)
            for idx, (a, bb, c) in enumerate(zero_samples):
                if bb == b:
                    grad += cur_U[a, :] * cur_W[c, :] * pred_zero[idx]
            cur_V[b, :] -= lr * grad

        # Update W
        for c in range(c_dim):
            if time.time() > deadline - 5:
                break
            grad = np.zeros(target_rank)
            for idx, (a, b, cc) in enumerate(nonzero):
                if cc == c:
                    grad += cur_U[a, :] * cur_V[b, :] * (pred_nonzero[idx] - 1.0)
            for idx, (a, b, cc) in enumerate(zero_samples):
                if cc == c:
                    grad += cur_U[a, :] * cur_V[b, :] * pred_zero[idx]
            cur_W[c, :] -= lr * grad

        # Sparsity thresholding
        cur_U[np.abs(cur_U) < sparsity_thresh] = 0.0
        cur_V[np.abs(cur_V) < sparsity_thresh] = 0.0
        cur_W[np.abs(cur_W) < sparsity_thresh] = 0.0

        # Recompute loss
        pred_nonzero = []
        for a, b, c in nonzero:
            pred_nonzero.append(np.sum(cur_U[a, :] * cur_V[b, :] * cur_W[c, :]))
        pred_nonzero = np.array(pred_nonzero)
        pred_zero = []
        for a, b, c in zero_samples:
            pred_zero.append(np.sum(cur_U[a, :] * cur_V[b, :] * cur_W[c, :]))
        pred_zero = np.array(pred_zero)
        new_loss = np.sum((pred_nonzero - 1.0) ** 2) + np.sum(pred_zero ** 2)

        if new_loss < best_loss:
            best_loss = new_loss
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
            improved = True
        else:
            # Decay learning rate
            lr *= 0.7
            cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()

        if best_loss < 1e-8:
            break

    return best_U, best_V, best_W, best_loss


def _try_rank_reduction(n, m, p, target_rank, deadline, seed=0):
    """Attempt to find a rank-target_rank decomposition via ALS refinement."""
    U, V, W, loss = _als_refine(n, m, p, target_rank, deadline, seed)
    if loss < 1e-6:
        # Verify fully
        if _verify_full(U, V, W, n, m, p, tol=1e-8):
            return U, V, W, True
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120 - 15  # 105s safety margin

    # Start with the block decomposition with Strassen (rank 51)
    rank, U, V, W = _build_block_decomposition(n, m, p)

    if _verify_full(U, V, W, n, m, p):
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    else:
        # Fallback: schoolbook rank 54
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    # Try to reduce rank below best_rank
    # The theoretical lower bound for 3x3 * 3x6 is at least 3*3*3 = 27 (if we could
    # use rank-9 for 3x3*3x3 blocks), but Strassen gives 7 for 2x2*2x2.
    # Let's try ranks from best_rank-1 down to some minimum.
    min_rank_to_try = 40

    for target_rank in range(best_rank - 1, max(min_rank_to_try - 1, 27) - 1, -1):
        if time.time() > deadline - 10:
            break

        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break

            U_t, V_t, W_t, success = _try_rank_reduction(
                n, m, p, target_rank, deadline, seed=seed
            )

            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break  # Found a valid decomposition at this rank, try lower

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
