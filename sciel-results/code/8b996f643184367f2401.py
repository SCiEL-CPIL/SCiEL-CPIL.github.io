import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill


def _strassen_2x2_coeffs():
    alpha = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    return alpha, beta, recon


def _build_77_rank():
    """Kronecker/block-Strassen rank-77 decomposition of the 4x4 x 4x6 tensor."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    alpha, beta, recon = _strassen_2x2_coeffs()

    def w_idx(out_r, out_c):
        return out_c * n + out_r

    def u_idx(a_r, a_c):
        return a_r * m + a_c

    def v_idx(b_r, b_c):
        return b_r * p + b_c

    col = 0
    for s in range(7):
        # Part 1: inner Strassen for X_s(2x2) * Y_s[:,0:2](2x2)
        for si in range(7):
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    U[u_idx(ar * 2 + x_row, br * 2 + x_col), col] += ca * cr
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    V[v_idx(ac * 2 + y_row, bc * 3 + y_col), col] += cb * cc
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    out_r = a * 2 + za
                    out_c = b * 3 + zb
                    W[w_idx(out_r, out_c), col] += gamma * zgamma
            col += 1

        # Part 2: schoolbook for X_s(2x2) * Y_s[:,2](2x1)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    U[u_idx(ar * 2 + u, br * 2 + v), col] += cr
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    V[v_idx(ac * 2 + v, bc * 3 + 2), col] += cc
                for a, b, gamma in recon[s]:
                    out_r = a * 2 + u
                    out_c = b * 3 + 2
                    W[w_idx(out_r, out_c), col] += gamma
                col += 1

    assert col == R, f"col={col} != R={R}"
    return U, V, W


def _build_schoolbook():
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W


def _recon_tensor(U, V, W, n, m, p):
    R = U.shape[1]
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    T = np.einsum('ikr,krr,irn->ikn', Uf, Vf, Wf)
    return T


def _target_tensor(n, m, p):
    return np.ones((n, m, p))


def _verify(U, V, W, n, m, p, num_trials=4, seed=0):
    target = _target_tensor(n, m, p)
    T = _recon_tensor(U, V, W, n, m, p)
    if not np.allclose(T, target, atol=1e-6):
        return False
    rng = np.random.default_rng(seed)
    for _ in range(num_trials):
        A = rng.integers(-9, 9, (n, m)).astype(float)
        B = rng.integers(-9, 9, (m, p)).astype(float)
        C_exp = A @ B
        C_rec = np.einsum('ikr,krr,irn->in', U, V, W)
        if not np.allclose(C_rec, C_exp, atol=1e-7):
            return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_removals=30, time_budget=25.0):
    """Attempt to reduce rank by zeroing columns and checking if the tensor
    still reconstructs. Returns (U, V, W, success) where success is True if
    at least one column was removed."""
    R = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    target = _target_tensor(n, m, p)
    start = time.time()
    removed = 0

    # Try removing columns one at a time, starting from the last
    for col in range(R - 1, -1, -1):
        if time.time() - start > time_budget:
            break
        if removed >= max_removals:
            break

        # Check if this column is "redundant" by trying to zero it
        # and seeing if we can compensate, OR if it's already near-zero
        if np.allclose(U[:, col], 0, atol=1e-9) or \
           np.allclose(V[:, col], 0, atol=1e-9) or \
           np.allclose(W[:, col], 0, atol=1e-9):
            # Already zero, just skip
            continue

        # Try zeroing this column and check if tensor still works
        # This is a greedy heuristic: zero the column, check, undo if needed
        saved_U = U[:, col].copy()
        saved_V = V[:, col].copy()
        saved_W = W[:, col].copy()

        U[:, col] = 0
        V[:, col] = 0
        W[:, col] = 0

        T = _recon_tensor(U, V, W, n, m, p)
        if np.allclose(T, target, atol=1e-6):
            removed += 1
            # Keep it zeroed
        else:
            # Undo
            U[:, col] = saved_U
            V[:, col] = saved_V
            W[:, col] = saved_W

    if removed > 0:
        # Compact: remove all-zero columns
        mask = ~np.allclose(U, 0, atol=1e-9, axis=0) & \
               ~np.allclose(V, 0, atol=1e-9, axis=0) & \
               ~np.allclose(W, 0, atol=1e-9, axis=0)
        # Actually, a column is zero if ANY of U,V,W is all zero for that column
        mask = np.ones(R, dtype=bool)
        for col in range(R):
            if np.allclose(U[:, col], 0, atol=1e-9) or \
               np.allclose(V[:, col], 0, atol=1e-9) or \
               np.allclose(W[:, col], 0, atol=1e-9):
                mask[col] = False

        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]

        # Final verification
        if _verify(U, V, W, n, m, p, num_trials=2, seed=42):
            return U, V, W, True
        else:
            return U, V, W, False

    return U, V, W, False


def _coordinate_descent_reduce(U, V, W, n, m, p, time_budget=20.0):
    """Try to reduce rank via coordinate descent: perturb columns slightly
    and see if we can merge or eliminate terms."""
    # This is a more aggressive approach - try to find linear dependencies
    # among the rank-1 terms
    R = U.shape[1]
    if R <= 1:
        return U, V, W, False

    start = time.time()
    target = _target_tensor(n, m, p)

    # Build the tensor and check for near-dependencies
    # If two columns are nearly proportional, we might be able to merge them
    # For now, just try the simple zeroing approach with more attempts

    U = U.copy()
    V = V.copy()
    W = W.copy()
    removed = 0

    for iteration in range(10):
        if time.time() - start > time_budget:
            break

        for col in range(U.shape[1] - 1, -1, -1):
            if time.time() - start > time_budget:
                break

            saved_U = U[:, col].copy()
            saved_V = V[:, col].copy()
            saved_W = W[:, col].copy()

            U[:, col] = 0
            V[:, col] = 0
            W[:, col] = 0

            T = _recon_tensor(U, V, W, n, m, p)
            if np.allclose(T, target, atol=1e-6):
                removed += 1
            else:
                U[:, col] = saved_U
                V[:, col] = saved_V
                W[:, col] = saved_W

    if removed > 0:
        mask = np.ones(U.shape[1], dtype=bool)
        for col in range(U.shape[1]):
            if np.allclose(U[:, col], 0, atol=1e-9) or \
               np.allclose(V[:, col], 0, atol=1e-9) or \
               np.allclose(W[:, col], 0, atol=1e-9):
                mask[col] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
        if _verify(U, V, W, n, m, p, num_trials=2, seed=42):
            return U, V, W, True

    return U, V, W, False


def _to_list(mat):
    result = []
    for row in mat:
        new_row = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                new_row.append(int(round(x)))
            elif abs(2*x - round(2*x)) < 1e-9:
                new_row.append(round(2*x) / 2.0)
            else:
                new_row.append(float(x))
        result.append(new_row)
    return result


def construct():
    n, m, p = DIMENSIONS

    # 1) Build the rank-77 structured Strassen decomposition
    try:
        U, V, W = _build_77_rank()
        if _verify(U, V, W, n, m, p, num_trials=4, seed=0):
            # 2) Try to reduce rank via zeroing redundant columns
            if time.time() - START_TIME < TIME_LIMIT - 30:
                U2, V2, W2, success = _try_reduce_rank(U, V, W, n, m, p,
                                                       max_removals=20,
                                                       time_budget=20.0)
                if success and U2.shape[1] < U.shape[1]:
                    return {"U": _to_list(U2), "V": _to_list(V2), "W": _to_list(W2)}

            # 3) Try coordinate descent reduction
            if time.time() - START_TIME < TIME_LIMIT - 15:
                U3, V3, W3, success3 = _coordinate_descent_reduce(U, V, W, n, m, p,
                                                                  time_budget=15.0)
                if success3 and U3.shape[1] < U.shape[1]:
                    return {"U": _to_list(U3), "V": _to_list(V3), "W": _to_list(W3)}

            # 4) Return the original rank-77
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass

    # 5) Fallback: schoolbook (always valid, rank 96)
    U, V, W = _build_schoolbook()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
