import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _rank11_solution():
    """Known valid rank-11 exact decomposition for 2x2 * 2x3."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]


def _round_quarter(mat):
    return [[round(x * 4) / 4.0 for x in row] for row in mat]


def _round_eighth(mat):
    return [[round(x * 8) / 8.0 for x in row] for row in mat]


def _als_step_U(V, W, n, m, p, r):
    """Solve for U given V, W."""
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U


def _als_step_V(U, W, n, m, p, r):
    """Solve for V given U, W."""
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V


def _als_step_W(U, V, n, m, p, r):
    """Solve for W given U, V."""
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W


def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    """Run ALS from the given starting point for `iters` iterations."""
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W


def _structured_rank_reduction_search(n, m, p, deadline):
    """
    Structured rank-reduction: parameterize rank-10 factors as rank-11
    solution with one column removed plus a rank-1 perturbation.
    Uses a smooth objective (sum of squared residuals) minimized via ALS
    with multiple structured initializations.
    """
    r = 10
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)

    # Base: drop last column (index 10)
    U_base = U11[:, :r]
    V_base = V11[:, :r]
    W_base = W11[:, :r]

    def _residual(U, V, W):
        """Compute max absolute residual over all (i,j,k)."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res = abs(s - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None

    # Try dropping different columns from the rank-11 solution
    for drop_col in range(11):
        if time.time() > deadline:
            break
        U_try = np.delete(U11, drop_col, axis=1)
        V_try = np.delete(V11, drop_col, axis=1)
        W_try = np.delete(W11, drop_col, axis=1)

        # Convert to list format for ALS
        U_list = U_try.tolist()
        V_list = V_try.tolist()
        W_list = W_try.tolist()

        # Run ALS to optimize
        U_opt, V_opt, W_opt = _als_optimize(U_list, V_list, W_list, n, m, p, r, iters=60, deadline=deadline)

        # Check residual
        U_arr = np.array(U_opt)
        V_arr = np.array(V_opt)
        W_arr = np.array(W_opt)
        res = _residual(U_arr, V_arr, W_arr)

        if res < best_residual:
            best_residual = res
            best_U, best_V, best_W = U_opt, V_opt, W_opt

        # Also try with random perturbation added to the dropped-column base
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            U_pert = [[x + random.gauss(0, 0.3) for x in row] for row in U_try.tolist()]
            V_pert = [[x + random.gauss(0, 0.3) for x in row] for row in V_try.tolist()]
            W_pert = [[x + random.gauss(0, 0.3) for x in row] for row in W_try.tolist()]

            U_opt, V_opt, W_opt = _als_optimize(U_pert, V_pert, W_pert, n, m, p, r, iters=40, deadline=deadline)

            U_arr = np.array(U_opt)
            V_arr = np.array(V_opt)
            W_arr = np.array(W_opt)
            res = _residual(U_arr, V_arr, W_arr)

            if res < best_residual:
                best_residual = res
                best_U, best_V, best_W = U_opt, V_opt, W_opt

    # Also try pure random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        U_rand = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(n*m)]
        V_rand = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(m*p)]
        W_rand = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(p*n)]

        U_opt, V_opt, W_opt = _als_optimize(U_rand, V_rand, W_rand, n, m, p, r, iters=50, deadline=deadline)

        U_arr = np.array(U_opt)
        V_arr = np.array(V_opt)
        W_arr = np.array(W_opt)
        res = _residual(U_arr, V_arr, W_arr)

        if res < best_residual:
            best_residual = res
            best_U, best_V, best_W = U_opt, V_opt, W_opt

    # Final polish on best solution found
    if best_U is not None and time.time() < deadline:
        best_U, best_V, best_W = _als_optimize(best_U, best_V, best_W, n, m, p, r, iters=30, deadline=deadline)

    # Try rounding at multiple precisions
    if best_U is not None:
        for round_fn, name in [(_round_half, "half"), (_round_quarter, "quarter"), (_round_eighth, "eighth")]:
            Ur, Vr, Wr = round_fn(best_U), round_fn(best_V), round_fn(best_W)
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr

        # Also try: round to half, then check; if not, try small local search
        Ur, Vr, Wr = _round_half(best_U), _round_half(best_V), _round_half(best_W)
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr

    return None


def _try_rank10(deadline):
    """Attempt to find a rank-10 decomposition via structured search + ALS + rounding."""
    n, m, p = 2, 2, 3

    # First: structured rank-reduction approach
    result = _structured_rank_reduction_search(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback: broader random search
    r = 10
    sol11 = _rank11_solution()
    U11, V11, W11 = sol11["U"], sol11["V"], sol11["W"]
    U_warm = [row[:r] for row in U11]
    V_warm = [row[:r] for row in V11]
    W_warm = [row[:r] for row in W11]

    inits = [(U_warm, V_warm, W_warm)]
    for _ in range(30):
        if time.time() > deadline:
            break
        U_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
        V_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
        W_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
        inits.append((U_rand, V_rand, W_rand))

    for U0, V0, W0 in inits:
        if time.time() > deadline:
            break
        U, V, W = _als_optimize(U0, V0, W0, n, m, p, r, iters=40, deadline=deadline)
        U, V, W = _als_optimize(U, V, W, n, m, p, r, iters=20, deadline=deadline)

        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr

        Uq, Vq, Wq = _round_quarter(U), _round_quarter(V), _round_quarter(W)
        if _verify(Uq, Vq, Wq, n, m, p, tol=1e-9):
            return Uq, Vq, Wq

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Guaranteed fallback: rank-11 solution
    best = _rank11_solution()

    # Attempt rank 10
    if time.time() < deadline:
        result10 = _try_rank10(deadline)
        if result10 is not None:
            U, V, W = result10
            best = {"U": U, "V": V, "W": W}

    return best
# EVOLVE-BLOCK-END
