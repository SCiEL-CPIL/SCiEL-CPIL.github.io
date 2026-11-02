"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 145.0


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
    """Fallback: 2 Strassen blocks (rank 7 each) for cols 0-3, schoolbook (rank 4) for col 4."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        U[0][t_offset + 0] = 1
        U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1
        U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1
        U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1
        U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1
        U[3][t_offset + 6] = -1

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


def _build_block_init(rank, n=2, m=2, p=5):
    """
    Build a block-structured initialization for rank-r decomposition.
    Uses Strassen for cols 0-1, Strassen for cols 2-3, schoolbook for col 4.
    If rank < 18, we zero out the last (18-rank) columns.
    This gives a structured starting point that the optimizer can refine.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    if rank < 7:
        return U, V, W

    # Strassen block 1: cols 0,1 -> t_offset 0
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1

    V[0 * p + 0, 0] = 1
    V[1 * p + 1, 0] = 1
    V[0 * p + 0, 1] = 1
    V[0 * p + 1, 2] = 1
    V[1 * p + 1, 2] = -1
    V[1 * p + 0, 3] = 1
    V[0 * p + 0, 3] = -1
    V[1 * p + 1, 4] = 1
    V[0 * p + 0, 5] = 1
    V[0 * p + 1, 5] = 1
    V[1 * p + 0, 6] = 1
    V[1 * p + 1, 6] = 1

    W[0 * n + 0, 0] = 1
    W[1 * n + 1, 0] = 1
    W[0 * n + 1, 1] = 1
    W[1 * n + 1, 1] = -1
    W[1 * n + 0, 2] = 1
    W[1 * n + 1, 2] = 1
    W[0 * n + 0, 3] = 1
    W[0 * n + 1, 3] = 1
    W[0 * n + 0, 4] = -1
    W[1 * n + 0, 4] = 1
    W[1 * n + 1, 5] = 1
    W[0 * n + 0, 6] = 1

    if rank < 14:
        return U, V, W

    # Strassen block 2: cols 2,3 -> t_offset 7
    U[0, 7] = 1; U[3, 7] = 1
    U[2, 8] = 1; U[3, 8] = 1
    U[0, 9] = 1
    U[3, 10] = 1
    U[0, 11] = 1; U[1, 11] = 1
    U[2, 12] = 1; U[0, 12] = -1
    U[1, 13] = 1; U[3, 13] = -1

    V[0 * p + 2, 7] = 1
    V[1 * p + 3, 7] = 1
    V[0 * p + 2, 8] = 1
    V[0 * p + 3, 9] = 1
    V[1 * p + 3, 9] = -1
    V[1 * p + 2, 10] = 1
    V[0 * p + 2, 10] = -1
    V[1 * p + 3, 11] = 1
    V[0 * p + 2, 12] = 1
    V[0 * p + 3, 12] = 1
    V[1 * p + 2, 13] = 1
    V[1 * p + 3, 13] = 1

    W[2 * n + 0, 7] = 1
    W[3 * n + 1, 7] = 1
    W[2 * n + 1, 8] = 1
    W[3 * n + 1, 8] = -1
    W[3 * n + 0, 9] = 1
    W[3 * n + 1, 9] = 1
    W[2 * n + 0, 10] = 1
    W[2 * n + 1, 10] = 1
    W[2 * n + 0, 11] = -1
    W[3 * n + 0, 11] = 1
    W[3 * n + 1, 12] = 1
    W[2 * n + 0, 13] = 1

    if rank < 18:
        return U, V, W

    # Schoolbook for col 4 -> t_offset 14
    U[0, 14] = 1; V[0 * p + 4, 14] = 1; W[4 * n + 0, 14] = 1
    U[1, 15] = 1; V[1 * p + 4, 15] = 1; W[4 * n + 0, 15] = 1
    U[2, 16] = 1; V[0 * p + 4, 16] = 1; W[4 * n + 1, 16] = 1
    U[3, 17] = 1; V[1 * p + 4, 17] = 1; W[4 * n + 1, 17] = 1

    return U, V, W


def _compute_residuals(params, rank, n=2, m=2, p=5):
    """Compute all residuals for the tensor decomposition."""
    U = params[:n * m * rank].reshape((n * m, rank))
    V = params[n * m * rank: n * m * rank + m * p * rank].reshape((m * p, rank))
    W = params[n * m * rank + m * p * rank:].reshape((p * n, rank))

    residuals = np.zeros(n * m * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                    residuals[idx] = val - target
                    idx += 1
    return residuals


def _sse_error(params, rank, n=2, m=2, p=5):
    """Raw SSE error."""
    res = _compute_residuals(params, rank, n, m, p)
    return np.sum(res ** 2)


def _smooth_error(params, rank, n=2, m=2, p=5, lam=30.0):
    """
    Smooth surrogate: log-sum-exp of squared residuals + epigraph penalty on max|residual|.
    This is differentiable and guides L-BFGS-B toward driving all residuals to zero.
    """
    res = _compute_residuals(params, rank, n, m, p)
    sq = res ** 2

    max_sq = np.max(sq)
    if max_sq > 1e8:
        sse = np.sum(sq)
    else:
        shifted = sq - max_sq
        sse = max_sq + np.log(np.sum(np.exp(shifted)))

    max_abs = np.max(np.abs(res))
    epigraph = lam * max_abs ** 2

    return sse + epigraph


def _snap_to_half_int(arr):
    """Snap array values to nearest half-integer."""
    return np.round(arr * 2) / 2


def _check_exact(U, V, W, rank, n=2, m=2, p=5, tol=1e-6):
    """Check if decomposition is exact within tolerance."""
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                    if abs(val - target) > tol:
                        return False
    return True


def _polish(params, rank, n=2, m=2, p=5, tol=1e-12, max_iter=5):
    """High-precision polish using L-BFGS-B with very tight tolerance."""
    x0 = params.copy()
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        res = minimize(_sse_error, x0, args=(rank, n, m, p), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False})
        x0 = res.x
        if res.fun < tol:
            break
    return x0


def _try_global_optimization(target_rank, time_budget, n=2, m=2, p=5):
    """
    Attempt to find an exact rank-target_rank decomposition for the full 2x2x5 tensor.
    Uses block-structured initialization + smooth surrogate L-BFGS-B + polish.
    Returns (U, V, W) if found, else None.
    """
    size = n * m * target_rank + m * p * target_rank + p * n * target_rank

    best_solution = None
    best_error = float('inf')

    # Generate multiple initializations
    inits = []

    # Init 1: Block-structured (Strassen+Strassen+Schoolbook, truncated)
    U_b, V_b, W_b = _build_block_init(target_rank, n, m, p)
    inits.append(np.concatenate([U_b.flatten(), V_b.flatten(), W_b.flatten()]))

    # Init 2: Random
    np.random.seed(42)
    inits.append(np.random.uniform(-1.0, 1.0, size=size))

    # Init 3: Block + small random perturbation
    U_b2, V_b2, W_b2 = _build_block_init(target_rank, n, m, p)
    np.random.seed(123)
    pert = np.random.normal(0, 0.1, size=size)
    x3 = np.concatenate([U_b2.flatten(), V_b2.flatten(), W_b2.flatten()]) + pert
    inits.append(x3)

    # Init 4: Another random
    np.random.seed(777)
    inits.append(np.random.uniform(-2.0, 2.0, size=size))

    n_restarts = min(len(inits), 20)
    for restart in range(n_restarts):
        if time.time() > time_budget:
            break
        x0 = inits[restart % len(inits)].copy()

        # L-BFGS-B with smooth surrogate
        res = minimize(_smooth_error, x0, args=(target_rank, n, m, p, 30.0),
                       method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})

        x_best = res.x
        err = _sse_error(x_best, target_rank, n, m, p)

        if err < best_error:
            best_error = err
            best_solution = x_best.copy()

        # If close, try polish
        if err < 1e-4:
            x_polished = _polish(x_best, target_rank, n, m, p, tol=1e-12, max_iter=3)
            err_polished = _sse_error(x_polished, target_rank, n, m, p)

            if err_polished < best_error:
                best_error = err_polished
                best_solution = x_polished.copy()

            if err_polished < 1e-8:
                U_f = x_polished[:n * m * target_rank].reshape((n * m, target_rank))
                V_f = x_polished[n * m * target_rank: n * m * target_rank + m * p * target_rank].reshape((m * p, target_rank))
                W_f = x_polished[n * m * target_rank + m * p * target_rank:].reshape((p * n, target_rank))

                if _check_exact(U_f, V_f, W_f, target_rank, n, m, p, tol=1e-6):
                    U_r = _snap_to_half_int(U_f)
                    V_r = _snap_to_half_int(V_f)
                    W_r = _snap_to_half_int(W_f)
                    if _check_exact(U_r, V_r, W_r, target_rank, n, m, p, tol=1e-6):
                        return U_r, V_r, W_r
                    return U_f, V_f, W_f

    # Try Nelder-Mead from best solution if it's close
    if best_solution is not None and best_error < 1e-2 and time.time() < time_budget:
        res = minimize(_sse_error, best_solution, args=(target_rank, n, m, p),
                       method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
        x_best = res.x
        err = _sse_error(x_best, target_rank, n, m, p)

        if err < best_error:
            best_error = err
            best_solution = x_best.copy()

        if err < 1e-6:
            U_f = x_best[:n * m * target_rank].reshape((n * m, target_rank))
            V_f = x_best[n * m * target_rank: n * m * target_rank + m * p * target_rank].reshape((m * p, target_rank))
            W_f = x_best[n * m * target_rank + m * p * target_rank:].reshape((p * n, target_rank))

            if _check_exact(U_f, V_f, W_f, target_rank, n, m, p, tol=1e-6):
                U_r = _snap_to_half_int(U_f)
                V_r = _snap_to_half_int(V_f)
                W_r = _snap_to_half_int(W_f)
                if _check_exact(U_r, V_r, W_r, target_rank, n, m, p, tol=1e-6):
                    return U_r, V_r, W_r
                return U_f, V_f, W_f

    return None


def construct():
    n, m, p = DIMENSIONS

    # Try rank 17 first (best possible improvement over 18)
    if time.time() < DEADLINE - 30:
        result = _try_global_optimization(17, time_budget=DEADLINE - 30, n=n, m=m, p=p)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _verify_full(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}

    # Try rank 16
    if time.time() < DEADLINE - 15:
        result = _try_global_optimization(16, time_budget=DEADLINE - 15, n=n, m=m, p=p)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _verify_full(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to rank 18
    return _get_rank_18_solution(n, m, p)

# EVOLVE-BLOCK-END
