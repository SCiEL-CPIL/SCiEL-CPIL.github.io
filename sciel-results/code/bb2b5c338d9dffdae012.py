import time
import numpy as np

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 145.0


def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True


def _build_tensor(n=2, m=2, p=5):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _strassen_block(n=2, m=2, p=5, col0=0, col1=1):
    """Strassen decomposition for 2x2 x 2x2 sub-block at columns col0, col1."""
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    U[0, 0] = 1; U[3, 0] = 1
    V[col0, 0] = 1; V[p + col1, 0] = 1
    W[col0 * n, 0] = 1; W[(col1) * n + 1, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[col0, 1] = 1
    W[(col0) * n + 1, 1] = 1; W[(col1) * n + 1, 1] = -1

    U[0, 2] = 1
    V[col1, 2] = 1; V[p + col1, 2] = -1
    W[(col1) * n, 2] = 1; W[(col1) * n + 1, 2] = 1

    U[3, 3] = 1
    V[p + col0, 3] = 1; V[col0, 3] = -1
    W[(col0) * n, 3] = 1; W[(col0) * n + 1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[p + col1, 4] = 1
    W[(col0) * n, 4] = -1; W[(col1) * n, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[col0, 5] = 1; V[col1, 5] = 1
    W[(col1) * n + 1, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[p + col0, 6] = 1; V[p + col1, 6] = 1
    W[(col0) * n, 6] = 1

    return U, V, W


def _tensor_residual(U, V, W, T):
    return T - np.einsum('ur,vr,wr->uvw', U, V, W)


def _als_refine(U, V, W, T, R, n, m, p, max_iter=200, tol=1e-14, deadline=None):
    prev_err = np.inf
    best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), np.inf

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n * m):
            b = T[u].reshape(-1)
            sol, *_ = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol

        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m * p):
            b = T[:, v, :].reshape(-1)
            sol, *_ = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol

        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p * n):
            b = T[:, :, w].reshape(-1)
            sol, *_ = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol

        err = np.linalg.norm(_tensor_residual(U, V, W, T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # Polish
    if best_err < 1e-6 and deadline is not None and time.time() < deadline - 2:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        for it in range(80):
            if time.time() > deadline - 1:
                break
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n * m):
                b = T[u].reshape(-1)
                sol, *_ = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m * p):
                b = T[:, v, :].reshape(-1)
                sol, *_ = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p * n):
                b = T[:, :, w].reshape(-1)
                sol, *_ = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            err = np.linalg.norm(_tensor_residual(U, V, W, T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-15:
                break

    return best_U, best_V, best_W, best_err


def _rationalize(U, V, W, T, R, n, m, p):
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-5):
        return None

    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom

        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-4):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]

            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-8:
                            mat[i][j] = int(round(mat[i][j]))

            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    return None


def _structured_init_v2(n, m, p, R, strategy, trial, rng):
    """Diverse structured initializations."""
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Precompute various Strassen blocks
    S01 = _strassen_block(n, m, p, 0, 1)
    S23 = _strassen_block(n, m, p, 2, 3)
    S02 = _strassen_block(n, m, p, 0, 2)
    S13 = _strassen_block(n, m, p, 1, 3)
    S04 = _strassen_block(n, m, p, 0, 4)
    S14 = _strassen_block(n, m, p, 1, 4)
    S24 = _strassen_block(n, m, p, 2, 4)
    S34 = _strassen_block(n, m, p, 3, 4)

    blocks = [S01, S23, S02, S13, S04, S14, S24, S34]

    if strategy == 0:
        # Single Strassen + random
        U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]
        if R > 7:
            U[:, 7:] = rng.standard_normal((n * m, R - 7)) * 0.5
            V[:, 7:] = rng.standard_normal((m * p, R - 7)) * 0.5
            W[:, 7:] = rng.standard_normal((p * n, R - 7)) * 0.5
    elif strategy == 1:
        # Two Strassen blocks
        U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]
        if R >= 14:
            U[:, 7:14] = S23[0]; V[:, 7:14] = S23[1]; W[:, 7:14] = S23[2]
        if R > 14:
            U[:, 14:] = rng.standard_normal((n * m, R - 14)) * 0.3
            V[:, 14:] = rng.standard_normal((m * p, R - 14)) * 0.3
            W[:, 14:] = rng.standard_normal((p * n, R - 14)) * 0.3
    elif strategy == 2:
        # Different pair of Strassen
        U[:, :7] = S02[0]; V[:, :7] = S02[1]; W[:, :7] = S02[2]
        if R >= 14:
            U[:, 7:14] = S14[0]; V[:, 7:14] = S14[1]; W[:, 7:14] = S14[2]
        if R > 14:
            U[:, 14:] = rng.standard_normal((n * m, R - 14)) * 0.3
            V[:, 14:] = rng.standard_normal((m * p, R - 14)) * 0.3
            W[:, 14:] = rng.standard_normal((p * n, R - 14)) * 0.3
    elif strategy == 3:
        # Another pair
        U[:, :7] = S04[0]; V[:, :7] = S04[1]; W[:, :7] = S04[2]
        if R >= 14:
            U[:, 7:14] = S13[0]; V[:, 7:14] = S13[1]; W[:, 7:14] = S13[2]
        if R > 14:
            U[:, 14:] = rng.standard_normal((n * m, R - 14)) * 0.3
            V[:, 14:] = rng.standard_normal((m * p, R - 14)) * 0.3
            W[:, 14:] = rng.standard_normal((p * n, R - 14)) * 0.3
    elif strategy == 4:
        # Random
        U = rng.standard_normal((n * m, R)) * 0.5
        V = rng.standard_normal((m * p, R)) * 0.5
        W = rng.standard_normal((p * n, R)) * 0.5
    elif strategy == 5:
        # Strassen + structured sparse
        U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]
        if R > 7:
            for r in range(7, R):
                # Add sparse structured terms
                idx_u = rng.integers(0, n * m)
                idx_v = rng.integers(0, m * p)
                idx_w = rng.integers(0, p * n)
                U[idx_u, r] = rng.choice([-1, 1])
                V[idx_v, r] = rng.choice([-1, 1])
                W[idx_w, r] = rng.choice([-1, 1])
    elif strategy == 6:
        # Three-way split: 2x2 + 2x2 + 2x1
        U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]
        if R >= 14:
            U[:, 7:14] = S24[0]; V[:, 7:14] = S24[1]; W[:, 7:14] = S24[2]
        if R > 14:
            U[:, 14:] = rng.standard_normal((n * m, R - 14)) * 0.3
            V[:, 14:] = rng.standard_normal((m * p, R - 14)) * 0.3
            W[:, 14:] = rng.standard_normal((p * n, R - 14)) * 0.3
    else:
        # Hybrid: partial Strassen + random
        U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]
        if R > 7:
            U[:, 7:] = rng.standard_normal((n * m, R - 7)) * 0.4
            V[:, 7:] = rng.standard_normal((m * p, R - 7)) * 0.4
            W[:, 7:] = rng.standard_normal((p * n, R - 7)) * 0.4
            # Add some structure
            for r in range(7, min(R, 12)):
                U[0, r] += 0.5
                V[4, r] += 0.5
                W[0, r] += 0.5

    return U, V, W


def _search_rank(n, m, p, R, timeout=20, n_trials=100, seed_base=0):
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)
    rng = np.random.default_rng(seed_base + R * 1000)

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        strategy = trial % 8
        U, V, W = _structured_init_v2(n, m, p, R, strategy, trial, rng)

        max_iter = 180 if trial < 30 else 120
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p,
                                    max_iter=max_iter, deadline=deadline)

        if err < 1e-8:
            result = _rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result

    return None


def _construct_rank18(n=2, m=2, p=5):
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    S01 = _strassen_block(n, m, p, 0, 1)
    U[:, :7] = S01[0]; V[:, :7] = S01[1]; W[:, :7] = S01[2]

    S23 = _strassen_block(n, m, p, 2, 3)
    U[:, 7:14] = S23[0]; V[:, 7:14] = S23[1]; W[:, 7:14] = S23[2]

    # 2x1 block for column 4
    U[0, 14] = 1; V[0 * p + 4, 14] = 1; W[4 * n + 0, 14] = 1
    U[1, 15] = 1; V[1 * p + 4, 15] = 1; W[4 * n + 0, 15] = 1
    U[2, 16] = 1; V[0 * p + 4, 16] = 1; W[4 * n + 1, 16] = 1
    U[3, 17] = 1; V[1 * p + 4, 17] = 1; W[4 * n + 1, 17] = 1

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    if not _verify(U_list, V_list, W_list, n, m, p):
        raise ValueError("Rank-18 construction failed verification")

    return {"U": U_list, "V": V_list, "W": W_list}


def _try_perturb_from_rank18(n, m, p, target_rank, timeout=15, n_trials=100, seed_base=0):
    """Try to find a lower-rank decomposition by perturbing the rank-18 solution."""
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)
    rng = np.random.default_rng(seed_base + 99999)

    # Build full rank-18
    full_U = np.zeros((n * m, 18))
    full_V = np.zeros((m * p, 18))
    full_W = np.zeros((p * n, 18))

    S01 = _strassen_block(n, m, p, 0, 1)
    full_U[:, :7] = S01[0]; full_V[:, :7] = S01[1]; full_W[:, :7] = S01[2]

    S23 = _strassen_block(n, m, p, 2, 3)
    full_U[:, 7:14] = S23[0]; full_V[:, 7:14] = S23[1]; full_W[:, 7:14] = S23[2]

    full_U[0, 14] = 1; full_V[4, 14] = 1; full_W[0, 14] = 1
    full_U[1, 15] = 1; full_V[9, 15] = 1; full_W[0, 15] = 1
    full_U[2, 16] = 1; full_V[4, 16] = 1; full_W[1, 16] = 1
    full_U[3, 17] = 1; full_V[9, 17] = 1; full_W[1, 17] = 1

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))

        mode = trial % 4
        if mode == 0:
            # Random subset of rank-18 terms
            indices = rng.choice(18, target_rank, replace=False)
            U[:] = full_U[:, indices]
            V[:] = full_V[:, indices]
            W[:] = full_W[:, indices]
        elif mode == 1:
            # First target_rank terms + small perturbation
            U[:] = full_U[:, :target_rank]
            V[:] = full_V[:, :target_rank]
            W[:] = full_W[:, :target_rank]
            U += rng.standard_normal((n * m, target_rank)) * 0.1
            V += rng.standard_normal((m * p, target_rank)) * 0.1
            W += rng.standard_normal((p * n, target_rank)) * 0.1
        elif mode == 2:
            # Keep Strassen part, randomize the rest
            n_strassen = min(target_rank, 14)
            U[:, :n_strassen] = full_U[:, :n_strassen]
            V[:, :n_strassen] = full_V[:, :n_strassen]
            W[:, :n_strassen] = full_W[:, :n_strassen]
            if target_rank > n_strassen:
                U[:, n_strassen:] = rng.standard_normal((n * m, target_rank - n_strassen)) * 0.5
                V[:, n_strassen:] = rng.standard_normal((m * p, target_rank - n_strassen)) * 0.5
                W[:, n_strassen:] = rng.standard_normal((p * n, target_rank - n_strassen)) * 0.5
        else:
            # Keep first Strassen, randomize rest
            n_s = min(target_rank, 7)
            U[:, :n_s] = full_U[:, :n_s]
            V[:, :n_s] = full_V[:, :n_s]
            W[:, :n_s] = full_W[:, :n_s]
            if target_rank > n_s:
                U[:, n_s:] = rng.standard_normal((n * m, target_rank - n_s)) * 0.5
                V[:, n_s:] = rng.standard_normal((m * p, target_rank - n_s)) * 0.5
                W[:, n_s:] = rng.standard_normal((p * n, target_rank - n_s)) * 0.5

        U, V, W, err = _als_refine(U, V, W, T, target_rank, n, m, p,
                                    max_iter=100, deadline=deadline)

        if err < 1e-8:
            result = _rationalize(U, V, W, T, target_rank, n, m, p)
            if result is not None:
                return result

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Try Rank 14
    remaining = deadline - time.time() - 35
    if remaining > 10:
        t14_budget = min(20, remaining)
        result = _search_rank(n, m, p, 14, timeout=t14_budget, n_trials=100, seed_base=1)
        if result is not None:
            return result

    # Try Rank 15
    remaining = deadline - time.time() - 30
    if remaining > 10:
        t15_budget = min(18, remaining)
        result = _search_rank(n, m, p, 15, timeout=t15_budget, n_trials=120, seed_base=2)
        if result is not None:
            return result

    # Try Rank 16
    remaining = deadline - time.time() - 25
    if remaining > 10:
        t16_budget = min(16, remaining)
        result = _search_rank(n, m, p, 16, timeout=t16_budget, n_trials=100, seed_base=3)
        if result is not None:
            return result

    # Try Rank 17 via perturbation from rank-18
    remaining = deadline - time.time() - 18
    if remaining > 8:
        t17_budget = min(14, remaining)
        result = _try_perturb_from_rank18(n, m, p, 17, timeout=t17_budget, n_trials=100, seed_base=4)
        if result is not None:
            return result

    # Try Rank 17 - standard search
    remaining = deadline - time.time() - 12
    if remaining > 6:
        t17_budget = min(10, remaining)
        result = _search_rank(n, m, p, 17, timeout=t17_budget, n_trials=80, seed_base=5)
        if result is not None:
            return result

    # Fallback Rank 18 (Fast and Guaranteed)
    return _construct_rank18(n, m, p)
