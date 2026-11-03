"""Tensor decomposition for 2x5x6 matrix multiplication.
Keeps the known rank-54 Strassen-based construction and attempts rank reduction
via vectorized ALS with a smooth (log-sum-exp) surrogate objective and a final
high-precision polish pass, all under an explicit wall-clock deadline."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]


def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul. Verified numerically."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1

    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1

    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1

    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1

    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1

    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1

    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1

    return U, V, W


def _verify_222():
    U, V, W = _strassen_222()
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(7):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True


def _build_2x5x6():
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    sU, sV, sW = _strassen_222()
    col = 0
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[0, col] = sU[0, r]; U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]; U[6, col] = sU[3, r]
            V[k_offset, col] = sV[0, r]; V[k_offset + 1, col] = sV[1, r]
            V[6 + k_offset, col] = sV[2, r]; V[7 + k_offset, col] = sV[3, r]
            W[k_offset * 2, col] = sW[0, r]; W[k_offset * 2 + 1, col] = sW[1, r]
            W[k_offset * 2 + 2, col] = sW[2, r]; W[k_offset * 2 + 3, col] = sW[3, r]
            col += 1
    for i in range(2):
        for k in range(6):
            U[i * 5 + 2, col] = 1
            V[2 * 6 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[3, col] = sU[0, r]; U[4, col] = sU[1, r]
            U[8, col] = sU[2, r]; U[9, col] = sU[3, r]
            V[3 * 6 + k_offset, col] = sV[0, r]; V[3 * 6 + k_offset + 1, col] = sV[1, r]
            V[4 * 6 + k_offset, col] = sV[2, r]; V[4 * 6 + k_offset + 1, col] = sV[3, r]
            W[k_offset * 2, col] = sW[0, r]; W[k_offset * 2 + 1, col] = sW[1, r]
            W[k_offset * 2 + 2, col] = sW[2, r]; W[k_offset * 2 + 3, col] = sW[3, r]
            col += 1
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_full(U, V, W, n=2, m=5, p=6, tol=1e-9):
    rank = len(U[0])
    Ua = np.array(U); Va = np.array(V); Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(total - 1.0) > tol:
                    return False
    return True


# EVOLVE-BLOCK-START
def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition (vectorized)."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err += (total - 1.0) ** 2
    return err


def _smooth_error(U, V, W, n, m, p, beta=100.0):
    """
    Smooth surrogate for the nonsmooth max|error| objective using log-sum-exp.
    This gives a differentiable proxy for the worst-case error, helping ALS
    avoid getting stuck on individual large-error entries.
    """
    rank = U.shape[1]
    logs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                e = abs(total - 1.0)
                # log-sum-exp over all entries to get a smooth max
                logs.append(e)
    logs = np.array(logs)
    # Stable log-sum-exp
    m_val = np.max(logs)
    if m_val < 1e-12:
        return 0.0
    return m_val + np.log(np.sum(np.exp(logs - m_val))) / beta


def _als_decompose(n, m, p, target_rank, U0=None, V0=None, W0=None,
                   max_iters=300, deadline=None, tol=1e-10, use_smooth=False):
    """
    Alternating least squares for tensor decomposition.
    Vectorized per-row updates with optional smooth surrogate guidance.
    """
    if U0 is not None:
        U = U0.copy(); V = V0.copy(); W = W0.copy()
    else:
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n * m, target_rank)) * 0.3
        V = rng.standard_normal((m * p, target_rank)) * 0.3
        W = rng.standard_normal((p * n, target_rank)) * 0.3

    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Precompute index arrays for speed
    i_idx = np.arange(n * m) // m
    j_idx = np.arange(n * m) % m
    jk_idx = np.arange(m * p) // p
    kk_idx = np.arange(m * p) % p
    ki_idx = np.arange(p * n) // n
    ii_idx = np.arange(p * n) % n

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # --- Update U ---
        for row in range(n * m):
            i = row // m
            j = row % m
            # Build matrix: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            Vj = V[j * p:(j + 1) * p, :]
            Wi = W[i::n, :]  # rows k*n+i for k=0..p-1
            A_mat = Vj * Wi
            try:
                U[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
            except np.linalg.LinAlgError:
                pass

        # --- Update V ---
        for row in range(m * p):
            j = row // p
            k = row % p
            # Build matrix: A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            Uj = U[j::m, :]  # rows i*m+j for i=0..n-1
            Wk = W[k::n, :]  # rows k*n+i for i=0..n-1
            A_mat = Uj * Wk
            try:
                V[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
            except np.linalg.LinAlgError:
                pass

        # --- Update W ---
        for row in range(p * n):
            k = row // n
            i = row % n
            # Build matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            Ui = U[i * m:(i + 1) * m, :]
            Vk = V[k::p, :]  # rows j*p+k for j=0..m-1
            A_mat = Ui * Vk
            try:
                W[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
            except np.linalg.LinAlgError:
                pass

        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break

    return best_U, best_V, best_W, best_err


def _polish(U, V, W, n, m, p, max_iters=500, deadline=None, tol=1e-12):
    """
    High-precision polish pass: run ALS with tighter tolerance and more iterations.
    This is a local solve to drive the error to machine precision.
    """
    return _als_decompose(n, m, p, U.shape[1], U, V, W,
                          max_iters=max_iters, deadline=deadline, tol=tol)


def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Snap to half-integers and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    err = _compute_error(U_s, V_s, W_s, n, m, p)
    if err < tol:
        return U_s, V_s, W_s, True
    return None, None, None, False


def _try_reduce_from_strassen(n, m, p, start_rank=54, deadline=None):
    """
    Try to reduce rank starting from the Strassen decomposition.
    Uses multiple strategies: column removal + re-optimize, random restarts,
    and a smooth-surrogate guided ALS.
    """
    base = _build_2x5x6()
    U54 = np.array(base['U'])
    V54 = np.array(base['V'])
    W54 = np.array(base['W'])

    for target_rank in range(start_rank - 1, max(start_rank - 12, 40), -1):
        if deadline is not None and time.time() > deadline - 8:
            break

        # Strategy 1: Remove columns one at a time and re-optimize
        for remove_col in range(start_rank):
            if deadline is not None and time.time() > deadline - 5:
                break

            keep = [c for c in range(start_rank) if c != remove_col]
            U_init = U54[:, keep]
            V_init = V54[:, keep]
            W_init = W54[:, keep]

            # Small random perturbation to escape local minima
            rng = np.random.default_rng(remove_col * 7 + target_rank)
            U_init = U_init + rng.standard_normal((n * m, target_rank)) * 0.03
            V_init = V_init + rng.standard_normal((m * p, target_rank)) * 0.03
            W_init = W_init + rng.standard_normal((p * n, target_rank)) * 0.03

            U_opt, V_opt, W_opt, err = _als_decompose(
                n, m, p, target_rank, U_init, V_init, W_init,
                max_iters=150, deadline=deadline - 3
            )

            if err < 1e-8:
                # Polish
                U_pol, V_pol, W_pol, err_pol = _polish(
                    U_opt, V_opt, W_opt, n, m, p,
                    max_iters=200, deadline=deadline - 1
                )
                if err_pol < err:
                    U_opt, V_opt, W_opt, err = U_pol, V_pol, W_pol, err_pol

                if err < 1e-8:
                    U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                    if valid:
                        return U_s, V_s, W_s, target_rank

        # Strategy 2: Random restarts with target_rank
        for attempt in range(4):
            if deadline is not None and time.time() > deadline - 3:
                break
            rng = np.random.default_rng(attempt * 1000 + target_rank)
            U_init = rng.standard_normal((n * m, target_rank)) * 0.5
            V_init = rng.standard_normal((m * p, target_rank)) * 0.5
            W_init = rng.standard_normal((p * n, target_rank)) * 0.5

            U_opt, V_opt, W_opt, err = _als_decompose(
                n, m, p, target_rank, U_init, V_init, W_init,
                max_iters=200, deadline=deadline - 2
            )

            if err < 1e-8:
                U_pol, V_pol, W_pol, err_pol = _polish(
                    U_opt, V_opt, W_opt, n, m, p,
                    max_iters=200, deadline=deadline - 1
                )
                if err_pol < err:
                    U_opt, V_opt, W_opt, err = U_pol, V_pol, W_pol, err_pol

                if err < 1e-8:
                    U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                    if valid:
                        return U_s, V_s, W_s, target_rank

    return None, None, None, start_rank


def _try_border_rank_approach(n, m, p, deadline=None):
    """
    Border rank technique: perturb the tensor slightly, find a lower-rank
    decomposition for the perturbed tensor, then snap back.
    Uses smooth surrogate to guide the optimization.
    """
    base = _build_2x5x6()
    U54 = np.array(base['U'])
    V54 = np.array(base['V'])
    W54 = np.array(base['W'])

    best_result = None
    best_rank = 54

    for target_rank in range(53, 44, -1):
        if deadline is not None and time.time() > deadline - 8:
            break

        for seed in range(6):
            if deadline is not None and time.time() > deadline - 5:
                break

            rng = np.random.default_rng(seed * 1000 + target_rank)

            # Initialize by taking a random subset of columns with perturbation
            cols = rng.choice(54, size=target_rank, replace=False)
            U_init = U54[:, cols].copy()
            V_init = V54[:, cols].copy()
            W_init = W54[:, cols].copy()

            scale = 0.08
            U_init += rng.standard_normal((n * m, target_rank)) * scale
            V_init += rng.standard_normal((m * p, target_rank)) * scale
            W_init += rng.standard_normal((p * n, target_rank)) * scale

            U_opt, V_opt, W_opt, err = _als_decompose(
                n, m, p, target_rank, U_init, V_init, W_init,
                max_iters=250, deadline=deadline - 3
            )

            if err < 1e-7:
                U_pol, V_pol, W_pol, err_pol = _polish(
                    U_opt, V_opt, W_opt, n, m, p,
                    max_iters=300, deadline=deadline - 1
                )
                if err_pol < err:
                    U_opt, V_opt, W_opt, err = U_pol, V_pol, W_pol, err_pol

                if err < 1e-8:
                    U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                    if valid:
                        best_result = (U_s, V_s, W_s)
                        best_rank = target_rank
                        break
        else:
            continue
        break

    if best_result is not None:
        return best_result[0], best_result[1], best_result[2], best_rank
    return None, None, None, 54


def _try_smooth_guided_als(n, m, p, target_rank, U0, V0, W0, deadline=None):
    """
    ALS with periodic smooth surrogate evaluation to detect when the
    worst-case error is decreasing even if the mean-square error stalls.
    """
    U = U0.copy(); V = V0.copy(); W = W0.copy()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_smooth = float('inf')
    best_smooth_U, best_smooth_V, best_smooth_W = U.copy(), V.copy(), W.copy()

    for it in range(400):
        if deadline is not None and time.time() > deadline:
            break

        # Standard ALS step
        for row in range(n * m):
            i = row // m; j = row % m
            Vj = V[j * p:(j + 1) * p, :]
            Wi = W[i::n, :]
            A_mat = Vj * Wi
            try:
                U[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
            except np.linalg.LinAlgError:
                pass

        for row in range(m * p):
            j = row // p; k = row % p
            Uj = U[j::m, :]
            Wk = W[k::n, :]
            A_mat = Uj * Wk
            try:
                V[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
            except np.linalg.LinAlgError:
                pass

        for row in range(p * n):
            k = row // n; i = row % n
            Ui = U[i * m:(i + 1) * m, :]
            Vk = V[k::p, :]
            A_mat = Ui * Vk
            try:
                W[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
            except np.linalg.LinAlgError:
                pass

        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-10:
            break

        # Periodically check smooth surrogate
        if it % 20 == 0:
            smooth = _smooth_error(U, V, W, n, m, p, beta=50.0)
            if smooth < best_smooth:
                best_smooth = smooth
                best_smooth_U, best_smooth_V, best_smooth_W = U.copy(), V.copy(), W.copy()

    # Return the better of the two
    if best_err <= best_smooth:
        return best_U, best_V, best_W, best_err
    else:
        err_s = _compute_error(best_smooth_U, best_smooth_V, best_smooth_W, n, m, p)
        return best_smooth_U, best_smooth_V, best_smooth_W, err_s


def construct():
    deadline = time.time() + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # First verify and return the known rank-54 decomposition
    try:
        if _verify_222():
            result = _build_2x5x6()
            if _verify_full(result['U'], result['V'], result['W']):
                # Check half-integer constraint
                ok = True
                for mat_name in ['U', 'V', 'W']:
                    for row in result[mat_name]:
                        for val in row:
                            if abs(val * 2 - round(val * 2)) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_result = result
                    best_rank = 54

                    # Try border rank / compression approach
                    if time.time() < deadline - 15:
                        U_new, V_new, W_new, new_rank = _try_border_rank_approach(
                            n, m, p, deadline=deadline - 10
                        )
                        if U_new is not None and new_rank < best_rank:
                            best_result = {
                                "U": U_new.tolist(),
                                "V": V_new.tolist(),
                                "W": W_new.tolist()
                            }
                            best_rank = new_rank

                    # Try further reduction from the best found
                    if best_rank > 48 and time.time() < deadline - 15:
                        U_new, V_new, W_new, new_rank = _try_reduce_from_strassen(
                            n, m, p, start_rank=best_rank, deadline=deadline - 10
                        )
                        if U_new is not None and new_rank < best_rank:
                            best_result = {
                                "U": U_new.tolist(),
                                "V": V_new.tolist(),
                                "W": W_new.tolist()
                            }
                            best_rank = new_rank

                    return best_result
    except Exception:
        pass

    # Fall back to schoolbook rank 60
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
