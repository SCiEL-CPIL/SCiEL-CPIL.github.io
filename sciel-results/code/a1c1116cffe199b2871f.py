import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)

    # Precompute triples for fast checking
    triples = np.array([(i * m + j, j * p + k, k * n + i)
                        for i in range(n) for j in range(m) for k in range(p)])

    # Try to find rank-43 by removing one column and re-solving
    for col in range(44):
        if time.time() > deadline - 10:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline, triples)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}


def build_rank44(n, m, p):
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return U, V, W


def check(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > 1e-6:
                    return False
    return True


def vectorized_check(U, V, W, triples):
    """Check all 48 tensor entries at once. Returns array of 48 values."""
    n, m, p = DIMENSIONS
    a_idx = triples[:, 0]
    b_idx = triples[:, 1]
    c_idx = triples[:, 2]

    U_sel = U[a_idx]      # (48, R)
    V_sel = V[b_idx]      # (48, R)
    W_sel = W[c_idx]      # (48, R)

    # Element-wise product then sum over rank
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    return vals


def smooth_minmax_loss(U, V, W, triples, beta=100.0):
    """Smooth min-max objective using log-sum-exp on squared errors."""
    vals = vectorized_check(U, V, W, triples)
    errs = (vals - 1.0) ** 2
    # Log-sum-exp for numerical stability
    max_err = np.max(errs)
    if max_err < 1e-12:
        return 0.0
    log_sum_exp = max_err + np.log(np.sum(np.exp(errs - max_err)))
    return log_sum_exp / beta


def smooth_minmax_grad(U, V, W, triples, beta=100.0):
    """Gradient of smooth min-max loss w.r.t. U, V, W."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    a_idx = triples[:, 0]
    b_idx = triples[:, 1]
    c_idx = triples[:, 2]

    U_sel = U[a_idx]
    V_sel = V[b_idx]
    W_sel = W[c_idx]

    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    errs = vals - 1.0
    sq_errs = errs ** 2

    # Soft-max weights
    max_sq = np.max(sq_errs)
    if max_sq < 1e-12:
        return np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    log_weights = sq_errs - max_sq
    weights = np.exp(log_weights) / np.sum(np.exp(log_weights))

    # d(loss)/d(vals) = 2 * weights * errs / beta
    d_vals = 2.0 * weights * errs / beta

    # d(vals)/d(U[a]) = V[b] * W[c] (element-wise, for each rank)
    # d(loss)/d(U) = sum over triples where a matches
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)

    for t in range(len(triples)):
        a, b, c = triples[t]
        dw = d_vals[t]
        dU[a] += dw * V_sel[t] * W_sel[t]
        dV[b] += dw * U_sel[t] * W_sel[t]
        dW[c] += dw * U_sel[t] * V_sel[t]

    return dU, dV, dW


def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline, triples):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    nU = n * m
    nV = m * p
    nW = p * n

    # Phase 1: Gradient-based optimization with smooth min-max objective
    lr = 0.1
    beta = 50.0
    max_iters = 2000
    patience = 100
    best_loss = smooth_minmax_loss(U43, V43, W43, triples, beta)
    best_U = U43.copy()
    best_V = V43.copy()
    best_W = W43.copy()
    no_improve = 0

    for it in range(max_iters):
        if time.time() > deadline - 20:
            break

        dU, dV, dW = smooth_minmax_grad(U43, V43, W43, triples, beta)

        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            cur_loss = smooth_minmax_loss(U43, V43, W43, triples, beta)
            if cur_loss < best_loss - 1e-8:
                best_loss = cur_loss
                best_U = U43.copy()
                best_V = V43.copy()
                best_W = W43.copy()
                no_improve = 0
            else:
                no_improve += 50
                lr *= 0.5
                beta = max(10.0, beta * 0.8)
            if no_improve > patience:
                break

        U43 -= lr * dU
        V43 -= lr * dV
        W43 -= lr * dW

        # Periodic check
        if it % 100 == 0:
            vals = vectorized_check(U43, V43, W43, triples)
            if np.max(np.abs(vals - 1)) < 1e-8:
                break

    # Use best found
    U43, V43, W43 = best_U, best_V, best_W

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2

    vals = vectorized_check(U43, V43, W43, triples)
    if np.max(np.abs(vals - 1)) < 1e-6:
        return U43, V43, W43

    # Phase 2: Linear algebraic rank-reduction test
    # Construct the 48xR coefficient matrix and check if we can find
    # a better decomposition by exploring the null space
    if time.time() < deadline - 5:
        result = linear_algebraic_reduction(U43, V43, W43, n, m, p, deadline, triples)
        if result is not None:
            return result

    # Phase 3: Local search with structured moves
    result = local_search(U43, V43, W43, n, m, p, deadline, triples)
    if result is not None:
        return result

    return None


def linear_algebraic_reduction(U, V, W, n, m, p, deadline, triples):
    """Try to find a rank-43 solution by exploiting linear dependencies."""
    R = U.shape[1]
    nU = n * m
    nV = m * p
    nW = p * n

    # For each row of U, build the "coefficient vector" in the space of
    # products V[b]*W[c]. If two rows have proportional coefficient vectors,
    # we might be able to merge them.
    # More practically: try adding a small perturbation and re-optimizing

    # Try: for each pair of rows in one matrix, see if a linear combination
    # of their "shadow" vectors (in the other two matrices) is simpler
    # This is expensive, so we do a limited search

    # Alternative: try random restarts with the gradient method
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 3:
            break

        # Perturb the current solution
        U_r = U + np.random.randn(nU, R) * 0.1
        V_r = V + np.random.randn(nV, R) * 0.1
        W_r = W + np.random.randn(nW, R) * 0.1

        # Quick gradient descent
        lr = 0.05
        for it in range(300):
            if time.time() > deadline - 3:
                break
            dU, dV, dW = smooth_minmax_grad(U_r, V_r, W_r, triples, beta=30.0)
            U_r -= lr * dU
            V_r -= lr * dV
            W_r -= lr * dW
            if it % 50 == 0:
                lr *= 0.9

        U_r = np.round(U_r * 2) / 2
        V_r = np.round(V_r * 2) / 2
        W_r = np.round(W_r * 2) / 2

        vals = vectorized_check(U_r, V_r, W_r, triples)
        if np.max(np.abs(vals - 1)) < 1e-6:
            return U_r, V_r, W_r

    return None


def local_search(U, V, W, n, m, p, deadline, triples):
    """Greedy local search trying individual half-integer flips."""
    R = U.shape[1]
    nU = n * m
    nV = m * p
    nW = p * n

    # Precompute which triples each row participates in
    U_part = [np.where(triples[:, 0] == a)[0] for a in range(nU)]
    V_part = [np.where(triples[:, 1] == b)[0] for b in range(nV)]
    W_part = [np.where(triples[:, 2] == c)[0] for c in range(nW)]

    def fast_check_partial(U, V, W, tri_indices):
        if len(tri_indices) == 0:
            return True
        a_idx = triples[tri_indices, 0]
        b_idx = triples[tri_indices, 1]
        c_idx = triples[tri_indices, 2]
        U_sel = U[a_idx]
        V_sel = V[b_idx]
        W_sel = W[c_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1)) < 1e-6

    vals = vectorized_check(U, V, W, triples)
    max_err = np.max(np.abs(vals - 1))

    for ls_iter in range(500):
        if time.time() > deadline - 2:
            break

        if max_err < 1e-6:
            return U, V, W

        # Find the most violated triple
        worst_tri = np.argmax(np.abs(vals - 1))
        a, b, c = triples[worst_tri]

        # Try flipping entries: prioritize the worst triple's rows
        candidates = []

        # U row a - try setting to 0, 0.5, -0.5, 1, -1
        for r in range(R):
            for target in [0.0, 0.5, -0.5, 1.0, -1.0]:
                if abs(U[a, r] - target) > 1e-9:
                    candidates.append(('U', a, r, target))

        # V row b
        for r in range(R):
            for target in [0.0, 0.5, -0.5, 1.0, -1.0]:
                if abs(V[b, r] - target) > 1e-9:
                    candidates.append(('V', b, r, target))

        # W row c
        for r in range(R):
            for target in [0.0, 0.5, -0.5, 1.0, -1.0]:
                if abs(W[c, r] - target) > 1e-9:
                    candidates.append(('W', c, r, target))

        # Also try some random rows
        for _ in range(5):
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = np.random.randint(nU)
                col_r = np.random.randint(R)
                target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                candidates.append(('U', row, col_r, target))
            elif mat == 'V':
                row = np.random.randint(nV)
                col_r = np.random.randint(R)
                target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                candidates.append(('V', row, col_r, target))
            else:
                row = np.random.randint(nW)
                col_r = np.random.randint(R)
                target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                candidates.append(('W', row, col_r, target))

        np.random.shuffle(candidates)

        improved = False
        for mat_name, row, col_r, target in candidates:
            if mat_name == 'U':
                old = U[row, col_r]
                U[row, col_r] = target
                if fast_check_partial(U, V, W, U_part[row]):
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < 1e-6:
                        return U, V, W
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                U[row, col_r] = old
            elif mat_name == 'V':
                old = V[row, col_r]
                V[row, col_r] = target
                if fast_check_partial(U, V, W, V_part[row]):
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < 1e-6:
                        return U, V, W
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                V[row, col_r] = old
            else:  # W
                old = W[row, col_r]
                W[row, col_r] = target
                if fast_check_partial(U, V, W, W_part[row]):
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < 1e-6:
                        return U, V, W
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                W[row, col_r] = old

        if not improved:
            # Try a broader random search
            for _ in range(10):
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = np.random.randint(nU)
                    col_r = np.random.randint(R)
                    target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                    old = U[row, col_r]
                    U[row, col_r] = target
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                    U[row, col_r] = old
                elif mat == 'V':
                    row = np.random.randint(nV)
                    col_r = np.random.randint(R)
                    target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                    old = V[row, col_r]
                    V[row, col_r] = target
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                    V[row, col_r] = old
                else:
                    row = np.random.randint(nW)
                    col_r = np.random.randint(R)
                    target = np.random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                    old = W[row, col_r]
                    W[row, col_r] = target
                    vals_new = vectorized_check(U, V, W, triples)
                    new_max_err = np.max(np.abs(vals_new - 1))
                    if new_max_err < max_err - 1e-9:
                        max_err = new_max_err
                        vals = vals_new
                        improved = True
                        break
                    W[row, col_r] = old

            if not improved:
                break

    return None
