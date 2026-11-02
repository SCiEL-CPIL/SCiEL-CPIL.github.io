"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Constructive rank-18 baseline (2 Strassen blocks + schoolbook for col 4).
2. Aggressive rank-16 search:
   - Structured sharing between Strassen blocks
   - Numerical ALS with rational snap
3. Rank-17 search as intermediate.
4. Fallbacks.

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    n, m, p = DIMENSIONS
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    R = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _snap_to_halfint(x, tol=1e-6):
    hx = round(x * 2)
    if abs(x - hx / 2.0) < tol:
        return float(hx) / 2.0
    return None


def _check_exact(U, V, W, T, tol=1e-6):
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < tol:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(128)
        if abs(x - float(f)) < tol:
            return f
        return None

    Uf = []
    for row in U:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Uf.append(r)
    Vf = []
    for row in V:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Vf.append(r)
    Wf = []
    for row in W:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Wf.append(r)

    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1

    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1

    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1

    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1

    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    r = 14
    U[0][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 15
    U[1][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 16
    U[2][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+1][r] = 1.0
    r = 17
    U[3][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+1][r] = 1.0

    return U, V, W


def _build_rank16_candidate():
    """Try to construct a rank-16 solution by sharing terms between blocks."""
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Block 1: columns 0,1 (7 terms, indices 0-6)
    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)

    # Block 2: columns 2,3 (7 terms, indices 7-13)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    # Column 4: need 4 terms but only 2 slots (14, 15)
    # Try to use Strassen-like structure for 2x2x1
    # Actually 2x2x1 needs rank 4 (schoolbook)
    # So rank 16 with this structure is impossible.
    # Return None to indicate failure
    return None


def _als_solve(U, V, W, T, R, max_iters=200, tol=1e-12, deadline=None):
    """Alternating least squares for tensor decomposition."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Update U
        for u in range(U_rows):
            A = np.zeros((V_rows * W_rows, R))
            for v in range(V_rows):
                for w in range(W_rows):
                    A[v * W_rows + w] = V[v] * W[w]
            b = T[u].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[u] = sol

        # Update V
        for v in range(V_rows):
            A = np.zeros((U_rows * W_rows, R))
            for u in range(U_rows):
                for w in range(W_rows):
                    A[u * W_rows + w] = U[u] * W[w]
            b = T[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[v] = sol

        # Update W
        for w in range(W_rows):
            A = np.zeros((U_rows * V_rows, R))
            for u in range(U_rows):
                for v in range(V_rows):
                    A[u * V_rows + v] = U[u] * V[v]
            b = T[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[w] = sol

        # Check convergence
        result = np.zeros((U_rows, V_rows, W_rows))
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(result - T)
        if err < tol:
            break

    return U, V, W


def _try_rank_als(R, T, deadline, n_seeds=8, max_iters=150):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    seeds = [42, 123, 7, 999, 2024, 55, 88, 13]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        for scale in [0.3, 1.0, 3.0]:
            if time.time() > deadline:
                break
            U0 = np.random.randn(U_rows, R) * scale
            V0 = np.random.randn(V_rows, R) * scale
            W0 = np.random.randn(W_rows, R) * scale

            U, V, W = U0.copy(), V0.copy(), W0.copy()
            U, V, W = _als_solve(U, V, W, T, R, max_iters=max_iters, deadline=deadline)

            if _check_decomposition(U, V, W, T, tol=1e-6):
                # Try to snap to half-integers
                for denom in [2, 1, 4, 8]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()

    return None


def _try_rank17_structured(T, deadline):
    """Try structured rank-17 by sharing one term between blocks."""
    n, m, p = DIMENSIONS
    U18, V18, W18 = _build_rank18()

    # Try removing each schoolbook term and compensating
    for remove_idx in [14, 15, 16, 17]:
        if time.time() > deadline:
            break

        U17 = [row[:] for row in U18]
        V17 = [row[:] for row in V18]
        W17 = [row[:] for row in W18]

        for i in range(n * m):
            U17[i][remove_idx] = 0.0
        for i in range(m * p):
            V17[i][remove_idx] = 0.0
        for i in range(p * n):
            W17[i][remove_idx] = 0.0

        # Try small modifications to other terms
        val_set = [-1.0, -0.5, 0.5, 1.0]
        best_err = np.inf
        best_candidate = None

        for modify_idx in range(14):
            if time.time() > deadline:
                break
            for du, dv, dw in product(val_set, repeat=3):
                if du == 0 and dv == 0 and dw == 0:
                    continue

                U_test = [row[:] for row in U17]
                V_test = [row[:] for row in V17]
                W_test = [row[:] for row in W17]

                for i in range(n * m):
                    if abs(U17[i][modify_idx]) > 1e-9:
                        U_test[i][modify_idx] += du
                for i in range(m * p):
                    if abs(V17[i][modify_idx]) > 1e-9:
                        V_test[i][modify_idx] += dv
                for i in range(p * n):
                    if abs(W17[i][modify_idx]) > 1e-9:
                        W_test[i][modify_idx] += dw

                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            total = 0.0
                            for r in range(17):
                                total += U_test[u][r] * V_test[v][r] * W_test[w][r]
                            err += (total - T[u, v, w]) ** 2

                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        best_candidate = (U_test, V_test, W_test)
                        break
            if best_err < 1e-10:
                break

        if best_err < 1e-10 and best_candidate is not None:
            return best_candidate

    return None


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    T = _build_target()

    # Baseline: rank-18
    U18, V18, W18 = _build_rank18()
    if _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Try rank-17
        result17 = _try_rank17_structured(T, deadline)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

        # Try rank-16 via ALS
        result16 = _try_rank_als(16, T, deadline, n_seeds=6, max_iters=100)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-6):
                return {"U": U16, "V": V16, "W": W16}

        # Try rank-17 via ALS
        result17_als = _try_rank_als(17, T, deadline, n_seeds=6, max_iters=100)
        if result17_als is not None:
            U17, V17, W17 = result17_als
            if _check_decomposition(U17, V17, W17, T, tol=1e-6):
                return {"U": U17, "V": V17, "W": W17}

        return {"U": U18, "V": V18, "W": W18}

    # Fallback: ALS for lower ranks
    for R in [16, 17, 18]:
        if time.time() > deadline:
            break
        result = _try_rank_als(R, T, deadline, n_seeds=4, max_iters=80)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # Final fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
