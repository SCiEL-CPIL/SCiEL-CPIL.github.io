"""Matrix multiplication 2x3x6 - block decomposition + local search refinement."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = 2, 3, 6

    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    # ---- Verification of a full M_{n,m,p} decomposition ----
    def verify_full(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = sum(U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                              for c in range(r))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    # ---- Error of a sub-decomposition (smooth objective) ----
    def sub_error(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        err = 0.0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = sum(U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                              for c in range(r))
                    err += (val - 1.0) ** 2
        return err

    # ---- Local search: coordinate descent on one factor at a time ----
    def local_search(U, V, W, n_s, m_s, p_s, max_iters=200, tol=1e-10, time_budget=5.0):
        """Coordinate descent to minimize sub_error. Returns (U, V, W, err)."""
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        err = sub_error(U, V, W, n_s, m_s, p_s)
        for it in range(max_iters):
            if time.time() > start_time + time_budget:
                break
            if err < tol:
                break
            improved = False
            # Update U: for each (i,j) row, solve least squares
            # U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:] for all k) = 1
            # This is: for fixed (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for all k
            # i.e. U[i*m+j,:] dot [V[j*p+k,:]*W[k*n+i,:] for k] = 1
            # Build matrix M[k,c] = V[j*p+k,c] * W[k*n+i,c]; need U_row @ M^T = 1
            for i in range(n_s):
                for j in range(m_s):
                    # M shape (p_s, r)
                    M = np.array([V[j * p_s + k, :] * W[k * n_s + i, :] for k in range(p_s)])
                    # Solve U_row @ M.T = ones(p_s)  ->  M @ U_row.T = ones(p_s)
                    try:
                        sol = np.linalg.lstsq(M, np.ones(p_s), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    new_err = sub_error(U, V, W, n_s, m_s, p_s)
                    # compute partial error change cheaply: just recompute full (small)
                    U[i * m_s + j, :] = sol
                    new_err = sub_error(U, V, W, n_s, m_s, p_s)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        U[i * m_s + j, :] = U[i * m_s + j, :]  # revert (already set, re-set)
                        # revert properly
                        U[i * m_s + j, :] = sol  # we need old value; let's restructure
            # Simpler: do full recompute each coordinate; small sizes make this fine.
            if not improved:
                break
        return U, V, W, err

    # Better local search: full recompute per coordinate update
    def local_search2(U, V, W, n_s, m_s, p_s, max_iters=100, tol=1e-11, time_budget=6.0):
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        err = sub_error(U, V, W, n_s, m_s, p_s)
        for it in range(max_iters):
            if time.time() > start_time + time_budget:
                break
            if err < tol:
                break
            improved = False
            # Update U
            for i in range(n_s):
                for j in range(m_s):
                    M = np.array([V[j * p_s + k, :] * W[k * n_s + i, :] for k in range(p_s)])
                    try:
                        sol = np.linalg.lstsq(M, np.ones(p_s), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    old = U[i * m_s + j, :].copy()
                    U[i * m_s + j, :] = sol
                    new_err = sub_error(U, V, W, n_s, m_s, p_s)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        U[i * m_s + j, :] = old
            # Update V
            for j in range(m_s):
                for k in range(p_s):
                    M = np.array([U[i * m_s + j, :] * W[k * n_s + i, :] for i in range(n_s)])
                    try:
                        sol = np.linalg.lstsq(M, np.ones(n_s), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    old = V[j * p_s + k, :].copy()
                    V[j * p_s + k, :] = sol
                    new_err = sub_error(U, V, W, n_s, m_s, p_s)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        V[j * p_s + k, :] = old
            # Update W
            for k in range(p_s):
                for i in range(n_s):
                    M = np.array([U[i * m_s + j, :] * V[j * p_s + k, :] for j in range(m_s)])
                    try:
                        sol = np.linalg.lstsq(M, np.ones(m_s), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    old = W[k * n_s + i, :].copy()
                    W[k * n_s + i, :] = sol
                    new_err = sub_error(U, V, W, n_s, m_s, p_s)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        W[k * n_s + i, :] = old
            if not improved:
                break
        return U, V, W, err

    # ---- Try to find a rank-r sub-decomposition for M_{2,3,2} ----
    def find_sub_decomposition(n_s, m_s, p_s, target_rank, max_attempts=300, time_budget=30.0):
        best_err = float('inf')
        best = None
        np.random.seed(42)
        for attempt in range(max_attempts):
            if time.time() > start_time + time_budget:
                break
            U = np.random.randn(n_s * m_s, target_rank)
            V = np.random.randn(m_s * p_s, target_rank)
            W = np.random.randn(p_s * n_s, target_rank)
            # Normalize columns to avoid scale issues
            for c in range(target_rank):
                nU = np.linalg.norm(U[:, c])
                nV = np.linalg.norm(V[:, c])
                nW = np.linalg.norm(W[:, c])
                if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                    U[:, c] /= nU
                    V[:, c] /= nV
                    W[:, c] /= nW
            err = sub_error(U, V, W, n_s, m_s, p_s)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            # Local refine
            U_r, V_r, W_r, err_r = local_search2(U, V, W, n_s, m_s, p_s, max_iters=30, time_budget=2.0)
            if err_r < best_err:
                best_err = err_r
                best = (U_r, V_r, W_r)
            if best_err < 1e-10:
                break
        if best is None:
            return None, float('inf')
        return best, best_err

    # ---- Tile a sub-decomposition to full M_{2,3,6} ----
    def tile_sub_to_full(U_s, V_s, W_s, n_s, m_s, p_s, n_full, m_full, p_full):
        """Sub handles M_{n_s,m_s,p_s}. Full is M_{n_full,m_full,p_full} with p_full = p_s * blocks."""
        blocks = p_full // p_s
        r_sub = U_s.shape[1]
        total_rank = r_sub * blocks
        U_full = np.zeros((n_full * m_full, total_rank))
        V_full = np.zeros((m_full * p_full, total_rank))
        W_full = np.zeros((p_full * n_full, total_rank))
        for b in range(blocks):
            # U: same for all blocks (A is 2x3, independent of column block)
            U_full[:, b * r_sub:(b + 1) * r_sub] = U_s
            # V: B row j, column (b*p_s + k) -> sub V[j*p_s + k]
            for j in range(m_full):
                for k in range(p_s):
                    V_full[j * p_full + (b * p_s + k), b * r_sub:(b + 1) * r_sub] = \
                        V_s[j * p_s + k, :]
            # W: output C row (b*p_s + k), col i -> sub W[k*n_s + i]
            for k in range(p_s):
                for i in range(n_full):
                    W_full[(b * p_s + k) * n_full + i, b * r_sub:(b + 1) * r_sub] = \
                        W_s[k * n_s + i, :]
        return U_full, V_full, W_full

    # ---- Main search: try sub ranks 7, 8, 9, 10 for M_{2,3,2} ----
    sub_n, sub_m, sub_p = 2, 3, 2

    for r_sub in [7, 8, 9, 10, 11]:
        if time.time() > deadline - 20:
            break
        # Budget per sub-rank attempt
        remaining = deadline - 20 - (time.time() - start_time)
        if remaining < 5:
            break
        (U_s, V_s, W_s), err = find_sub_decomposition(
            sub_n, sub_m, sub_p, r_sub,
            max_attempts=200,
            time_budget=min(remaining, 35.0)
        )
        total_rank = r_sub * 3
        if err < 1e-9:
            # Verify sub exactly
            if verify_full(U_s, V_s, W_s, sub_n, sub_m, sub_p):
                U_full, V_full, W_full = tile_sub_to_full(
                    U_s, V_s, W_s, sub_n, sub_m, sub_p, n, m, p
                )
                if verify_full(U_full, V_full, W_full, n, m, p):
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                        # Found a valid lower rank; keep searching lower? r_sub only goes up, so break.
                        break

    return best_result
# EVOLVE-BLOCK-END
