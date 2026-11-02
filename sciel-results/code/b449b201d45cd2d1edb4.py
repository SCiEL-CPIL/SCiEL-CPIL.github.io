"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 143.0


def _time_left():
    return DEADLINE - time.time()


def _verify_full(U, V, W, n=2, m=2, p=5, tol=1e-8):
    """Verify the full 2x2x5 tensor decomposition. Returns True if exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + i][c]
                    if abs(val - target) > tol:
                        return False
    return True


def _get_rank_18_solution(n=2, m=2, p=5):
    """Fast hardcoded rank-18: 2 Strassen blocks (7 each) + schoolbook (4) for col 4."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        U[0][t_offset + 0] = 1; U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1; U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1; U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1; U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1; U[3][t_offset + 6] = -1

        k0 = k_start
        k1 = k_start + 1
        V[0 * p + k0][t_offset + 0] = 1
        V[1 * p + k1][t_offset + 0] = 1
        V[0 * p + k0][t_offset + 1] = 1
        V[0 * p + k1][t_offset + 2] = 1
        V[1 * p + k1][t_offset + 2] = -1
        V[1 * p + k0][t_offset + 3] = 1
        V[0 * p + k0][t_offset + 3] = -1
        V[1 * p + k1][t_offset + 4] = 1
        V[0 * p + k0][t_offset + 5] = 1
        V[0 * p + k1][t_offset + 5] = 1
        V[1 * p + k0][t_offset + 6] = 1
        V[1 * p + k1][t_offset + 6] = 1

        W[k0 * n + 0][t_offset + 0] = 1
        W[k1 * n + 1][t_offset + 0] = 1
        W[k0 * n + 1][t_offset + 1] = 1
        W[k1 * n + 1][t_offset + 1] = -1
        W[k1 * n + 0][t_offset + 2] = 1
        W[k1 * n + 1][t_offset + 2] = 1
        W[k0 * n + 0][t_offset + 3] = 1
        W[k0 * n + 1][t_offset + 3] = 1
        W[k0 * n + 0][t_offset + 4] = -1
        W[k1 * n + 0][t_offset + 4] = 1
        W[k1 * n + 1][t_offset + 5] = 1
        W[k0 * n + 0][t_offset + 6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    t = 14
    U[0][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[1][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[2][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 1][t] = 1; t += 1
    U[3][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 1][t] = 1

    return {"U": U, "V": V, "W": W}


def _error_223(params, rank):
    """Raw SSE error for 2x2x3 sub-decomposition."""
    n_sub, m_sub, p_sub = 2, 2, 3
    U = params[:n_sub * m_sub * rank].reshape((n_sub * m_sub, rank))
    V = params[n_sub * m_sub * rank: n_sub * m_sub * rank + m_sub * p_sub * rank].reshape((m_sub * p_sub, rank))
    W = params[n_sub * m_sub * rank + m_sub * p_sub * rank:].reshape((p_sub * n_sub, rank))

    err = 0.0
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m_sub + j, c] * V[jp * p_sub + k, c] * W[k * n_sub + i, c]
                    err += (val - target) ** 2
    return err


def _smooth_error_223(params, rank, lam=50.0):
    """Smooth surrogate: log-sum-exp of squared residuals + epigraph penalty."""
    n_sub, m_sub, p_sub = 2, 2, 3
    U = params[:n_sub * m_sub * rank].reshape((n_sub * m_sub, rank))
    V = params[n_sub * m_sub * rank: n_sub * m_sub * rank + m_sub * p_sub * rank].reshape((m_sub * p_sub, rank))
    W = params[n_sub * m_sub * rank + m_sub * p_sub * rank:].reshape((p_sub * n_sub, rank))

    residuals = []
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m_sub + j, c] * V[jp * p_sub + k, c] * W[k * n_sub + i, c]
                    residuals.append(val - target)

    residuals = np.array(residuals)
    sq = residuals ** 2

    max_sq = np.max(sq)
    if max_sq > 1e6:
        sse = np.sum(sq)
    else:
        shifted = sq - max_sq
        sse = max_sq + np.log(np.sum(np.exp(shifted)))

    max_abs = np.max(np.abs(residuals))
    epigraph = lam * max_abs ** 2

    return sse + epigraph


def _check_exact_223(U, V, W, rank, tol=1e-6):
    """Check if 2x2x3 decomposition is exact within tolerance."""
    n_sub, m_sub, p_sub = 2, 2, 3
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m_sub + j, c] * V[jp * p_sub + k, c] * W[k * n_sub + i, c]
                    if abs(val - target) > tol:
                        return False
    return True


def _snap_to_half_int(U, V, W, rank):
    """Snap values to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r


def _polish_223(U, V, W, rank, tol=1e-12, max_iter=5):
    """High-precision polish with L-BFGS-B."""
    n_sub, m_sub, p_sub = 2, 2, 3
    size = n_sub * m_sub * rank + m_sub * p_sub * rank + p_sub * n_sub * rank
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        res = minimize(_error_223, x0, args=(rank,), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False})
        x0 = res.x
        if res.fun < tol:
            break

    U_p = x0[:n_sub * m_sub * rank].reshape((n_sub * m_sub, rank))
    V_p = x0[n_sub * m_sub * rank: n_sub * m_sub * rank + m_sub * p_sub * rank].reshape((m_sub * p_sub, rank))
    W_p = x0[n_sub * m_sub * rank + m_sub * p_sub * rank:].reshape((p_sub * n_sub, rank))
    return U_p, V_p, W_p


def _try_find_223_decomposition(target_rank, time_budget):
    """
    Attempt to find an exact rank-target_rank decomposition for 2x2x3.
    Multiple strategies: L-BFGS-B with smooth surrogate, Nelder-Mead polish.
    Returns (U, V, W) if found, else None.
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    size = n_sub * m_sub * target_rank + m_sub * p_sub * target_rank + p_sub * n_sub * target_rank

    best_solution = None
    best_error = float('inf')

    # Strategy 1: L-BFGS-B with smooth surrogate, many restarts with varied seeds
    n_restarts = 60
    for restart in range(n_restarts):
        if _time_left() < time_budget:
            break
        np.random.seed(restart * 31 + target_rank * 13 + 7)
        # Mix of uniform and structured initializations
        if restart % 3 == 0:
            x0 = np.random.uniform(-2.0, 2.0, size=size)
        elif restart % 3 == 1:
            x0 = np.random.uniform(-1.0, 1.0, size=size)
        else:
            # Sparse-like: most entries zero, a few random
            x0 = np.zeros(size)
            n_nonzero = max(5, size // 4)
            idx = np.random.choice(size, n_nonzero, replace=False)
            x0[idx] = np.random.uniform(-1.5, 1.5, n_nonzero)

        res = minimize(_smooth_error_223, x0, args=(target_rank, 50.0), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13, 'disp': False})

        x_best = res.x
        err = _error_223(x_best, target_rank)

        if err < best_error:
            best_error = err
            best_solution = x_best.copy()

        # If very close, try polish
        if err < 1e-3:
            U_t = x_best[:n_sub * m_sub * target_rank].reshape((n_sub * m_sub, target_rank))
            V_t = x_best[n_sub * m_sub * target_rank: n_sub * m_sub * target_rank + m_sub * p_sub * target_rank].reshape((m_sub * p_sub, target_rank))
            W_t = x_best[n_sub * m_sub * target_rank + m_sub * p_sub * target_rank:].reshape((p_sub * n_sub, target_rank))

            U_p, V_p, W_p = _polish_223(U_t, V_t, W_t, target_rank, tol=1e-13, max_iter=3)
            x_polished = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
            err_polished = _error_223(x_polished, target_rank)

            if err_polished < best_error:
                best_error = err_polished
                best_solution = x_polished.copy()

            if err_polished < 1e-8:
                U_final = x_polished[:n_sub * m_sub * target_rank].reshape((n_sub * m_sub, target_rank))
                V_final = x_polished[n_sub * m_sub * target_rank: n_sub * m_sub * target_rank + m_sub * p_sub * target_rank].reshape((m_sub * p_sub, target_rank))
                W_final = x_polished[n_sub * m_sub * target_rank + m_sub * p_sub * target_rank:].reshape((p_sub * n_sub, target_rank))

                if _check_exact_223(U_final, V_final, W_final, target_rank, tol=1e-6):
                    U_r, V_r, W_r = _snap_to_half_int(U_final, V_final, W_final, target_rank)
                    if _check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                        return (U_r, V_r, W_r)
                    return (U_final, V_final, W_final)

    # Strategy 2: Nelder-Mead on raw SSE from best solution
    if best_solution is not None and best_error < 1e-2 and _time_left() > time_budget:
        res = minimize(_error_223, best_solution, args=(target_rank,), method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
        x_best = res.x
        err = _error_223(x_best, target_rank)

        if err < best_error:
            best_error = err
            best_solution = x_best.copy()

        if err < 1e-6:
            U_final = x_best[:n_sub * m_sub * target_rank].reshape((n_sub * m_sub, target_rank))
            V_final = x_best[n_sub * m_sub * target_rank: n_sub * m_sub * target_rank + m_sub * p_sub * target_rank].reshape((m_sub * p_sub, target_rank))
            W_final = x_best[n_sub * m_sub * target_rank + m_sub * p_sub * target_rank:].reshape((p_sub * n_sub, target_rank))

            if _check_exact_223(U_final, V_final, W_final, target_rank, tol=1e-6):
                U_r, V_r, W_r = _snap_to_half_int(U_final, V_final, W_final, target_rank)
                if _check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                    return (U_r, V_r, W_r)
                return (U_final, V_final, W_final)

    return None


def _assemble_full(U_sub, V_sub, W_sub, rank_sub, n=2, m=2, p=5):
    """Assemble full 2x2x5 decomposition from Strassen block (cols 0-1) + sub-decomp (cols 2-4)."""
    total_rank = 7 + rank_sub
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen block for cols 0, 1
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1

    V[0 * p + 0][0] = 1
    V[1 * p + 1][0] = 1
    V[0 * p + 0][1] = 1
    V[0 * p + 1][2] = 1
    V[1 * p + 1][2] = -1
    V[1 * p + 0][3] = 1
    V[0 * p + 0][3] = -1
    V[1 * p + 1][4] = 1
    V[0 * p + 0][5] = 1
    V[0 * p + 1][5] = 1
    V[1 * p + 0][6] = 1
    V[1 * p + 1][6] = 1

    W[0 * n + 0][0] = 1
    W[1 * n + 1][0] = 1
    W[0 * n + 1][1] = 1
    W[1 * n + 1][1] = -1
    W[1 * n + 0][2] = 1
    W[1 * n + 1][2] = 1
    W[0 * n + 0][3] = 1
    W[0 * n + 1][3] = 1
    W[0 * n + 0][4] = -1
    W[1 * n + 0][4] = 1
    W[1 * n + 1][5] = 1
    W[0 * n + 0][6] = 1

    # Copy sub-decomposition (cols 2,3,4 of B)
    for i in range(2):
        for j in range(2):
            for c in range(rank_sub):
                U[i * m + j][7 + c] = U_sub[i * 2 + j, c]

    for j in range(2):
        for k_sub in range(3):
            k_global = 2 + k_sub
            for c in range(rank_sub):
                V[j * p + k_global][7 + c] = V_sub[j * 3 + k_sub, c]

    for k_sub in range(3):
        k_global = 2 + k_sub
        for i in range(2):
            for c in range(rank_sub):
                W[k_global * n + i][7 + c] = W_sub[k_sub * 2 + i, c]

    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS

    # First, get the guaranteed rank-18 solution (instant)
    rank_18_sol = _get_rank_18_solution(n, m, p)

    # Try to find rank 9 for 2x2x3 (total rank 16)
    sub_decomp = None
    sub_rank = None

    if _time_left() > 30:
        sub_decomp = _try_find_223_decomposition(9, time_budget=30)
        if sub_decomp is not None:
            sub_rank = 9

    # Try to find rank 10 for 2x2x3 (total rank 17)
    if sub_decomp is None and _time_left() > 20:
        sub_decomp = _try_find_223_decomposition(10, time_budget=20)
        if sub_decomp is not None:
            sub_rank = 10

    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]

        full_sol = _assemble_full(U_sub, V_sub, W_sub, rank_sub, n, m, p)

        # Verify the full solution
        if _verify_full(full_sol["U"], full_sol["V"], full_sol["W"], n, m, p, tol=1e-6):
            return full_sol
        else:
            return rank_18_sol
    else:
        return rank_18_sol

# EVOLVE-BLOCK-END
