import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    # Reconstruct full tensor
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    # Build target tensor
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    if np.linalg.norm(T_recon - T_target) > tol:
        return False
    return True


def _check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast check: only verify the 40 one-positions and a few zero positions."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    # Check all 40 one-positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.dot(U[idx_a, :], V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check a sample of zero-positions to catch gross errors
    # Full tensor size is 8*20*10 = 1600, 40 are ones, 1560 are zeros
    # Check all zeros for correctness
    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                is_one = (i == i2 and j == j2 and k == k2)
                if is_one:
                    continue
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p

    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook rank
    rank_sb = n * m * p

    # Prepare schoolbook fallback
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def snap_half_integer(mat):
        return np.round(mat * 2) / 2.0

    def try_rank(r, time_budget, max_restarts=15, max_iters=800):
        """Try to find a rank-r decomposition using random restarts + SVD greedy."""
        best_local = None
        best_err = np.inf

        for restart in range(max_restarts):
            if time.time() > time_budget:
                break

            # Random init with small magnitude
            scale = 0.5
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(np_, r) * scale

            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > time_budget:
                    break

                # Compute current reconstruction
                T_recon = np.zeros_like(T)
                for rr in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])

                err = np.linalg.norm(T - T_recon)
                if err < 1e-9:
                    best_local = (U.copy(), V.copy(), W.copy())
                    best_err = 0
                    return best_local

                if err < best_err:
                    best_err = err
                    best_local = (U.copy(), V.copy(), W.copy())

                if err > prev_err * (1 + 1e-6):
                    break
                prev_err = err

                # Update one column at a time (round-robin)
                idx = it % r
                # Residual without this term
                T_other = T - T_recon + np.einsum('a,b,c->abc', U[:, idx], V[:, idx], W[:, idx])

                # SVD-based rank-1 update
                Tf = T_other.reshape(nm, mp * np_)
                U_, S_, Vt_ = np.linalg.svd(Tf, full_matrices=False)
                if len(S_) == 0 or S_[0] < 1e-15:
                    continue
                u_vec = U_[:, 0] * np.sqrt(S_[0])
                v_vec_full = Vt_[0, :] * np.sqrt(S_[0])
                V_mat = v_vec_full.reshape(mp, np_)
                V2_, S2_, Wt_ = np.linalg.svd(V_mat, full_matrices=False)
                if len(S2_) == 0 or S2_[0] < 1e-15:
                    continue
                v2_vec = V2_[:, 0] * np.sqrt(S2_[0])
                w_vec = Wt_[0, :] * np.sqrt(S2_[0])

                U[:, idx] = u_vec
                V[:, idx] = v2_vec
                W[:, idx] = w_vec

        return best_local

    def polish(U, V, W, time_budget, max_iters=500):
        """High-precision polish: tighten the decomposition."""
        r = U.shape[1]
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > time_budget:
                break

            T_recon = np.zeros_like(T)
            for rr in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])

            err = np.linalg.norm(T - T_recon)
            if err < 1e-12:
                return U, V, W

            if err > prev_err * (1 + 1e-8):
                break
            prev_err = err

            idx = it % r
            T_other = T - T_recon + np.einsum('a,b,c->abc', U[:, idx], V[:, idx], W[:, idx])

            Tf = T_other.reshape(nm, mp * np_)
            U_, S_, Vt_ = np.linalg.svd(Tf, full_matrices=False)
            if len(S_) == 0 or S_[0] < 1e-15:
                continue
            u_vec = U_[:, 0] * np.sqrt(S_[0])
            v_vec_full = Vt_[0, :] * np.sqrt(S_[0])
            V_mat = v_vec_full.reshape(mp, np_)
            V2_, S2_, Wt_ = np.linalg.svd(V_mat, full_matrices=False)
            if len(S2_) == 0 or S2_[0] < 1e-15:
                continue
            v2_vec = V2_[:, 0] * np.sqrt(S2_[0])
            w_vec = Wt_[0, :] * np.sqrt(S2_[0])

            U[:, idx] = u_vec
            V[:, idx] = v2_vec
            W[:, idx] = w_vec

        return U, V, W

    # Try ranks from low to high
    # Lower bound is max(nm, mp, np_) = 20
    min_rank = max(nm, mp, np_)

    # Prioritize ranks that are known or likely good
    # For 2x4x5, try 20, 21, 22, ... up to schoolbook
    rank_order = list(range(min_rank, rank_sb + 1))

    best_result = None
    for r in rank_order:
        if time.time() > deadline - 10:
            break

        time_budget = min(deadline - 5, time.time() + 15)
        result = try_rank(r, time_budget)

        if result is not None:
            U, V, W = result

            # Try to snap to half-integers
            U_s = snap_half_integer(U)
            V_s = snap_half_integer(V)
            W_s = snap_half_integer(W)

            if _check_decomposition_fast(U_s, V_s, W_s, n, m, p):
                # Polish the snapped version
                U_s, V_s, W_s = polish(U_s, V_s, W_s, min(deadline - 5, time.time() + 5))
                if _check_decomposition_fast(U_s, V_s, W_s, n, m, p):
                    best_result = (U_s, V_s, W_s)
                    break

            # Check non-snapped
            if _check_decomposition_fast(U, V, W, n, m, p):
                U, V, W = polish(U, V, W, min(deadline - 5, time.time() + 5))
                if _check_decomposition_fast(U, V, W, n, m, p):
                    best_result = (U, V, W)
                    break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
