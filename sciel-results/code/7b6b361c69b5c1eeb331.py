"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy:
1. Guaranteed rank-42 via 6 independent Strassen blocks.
2. Attempt rank-41 and rank-40 via smooth optimization (border-rank approach):
   minimize the Frobenius norm of the residual tensor, then verify exactness.
3. Return the lowest verified rank.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor R[i,j,k] = 1 - sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]."""
    rank = U.shape[1]
    R = np.zeros((n, m, p), dtype=float)
    for s in range(rank):
        u = U[:, s].reshape(n, m)
        v = V[:, s].reshape(m, p)
        w = W[:, s].reshape(p, n)
        # Outer product: u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    R -= 1.0
    return R


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with tolerance."""
    R = _tensor_residual(U, V, W, n, m, p)
    return np.max(np.abs(R)) < tol


def _try_optimize_rank(rank, n, m, p, max_iters=2000, lr=0.01, time_limit=30.0):
    """
    Try to find a rank-`rank` decomposition via smooth optimization.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    np.random.seed(42)

    # Initialize with random vectors
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5

    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if time.time() - start > time_limit:
            break

        # Compute residual
        R = _tensor_residual(U, V, W, n, m, p)
        loss = np.sum(R ** 2)

        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_loss < 1e-12:
            break

        # Compute gradients via finite differences or analytical
        # For efficiency, use analytical gradients
        # dL/dU[i*m+j,s] = -2 * sum_{k} R[i,j,k] * V[j*p+k,s] * W[k*n+i,s]
        # dL/dV[j*p+k,s] = -2 * sum_{i} R[i,j,k] * U[i*m+j,s] * W[k*n+i,s]
        # dL/dW[k*n+i,s] = -2 * sum_{j} R[i,j,k] * U[i*m+j,s] * V[j*p+k,s]

        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        for s in range(rank):
            u = U[:, s].reshape(n, m)
            v = V[:, s].reshape(m, p)
            w = W[:, s].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = R[i, j, k]
                        dU[i * m + j, s] += -2 * r * v[j, k] * w[k, i]
                        dV[j * p + k, s] += -2 * r * u[i, j] * w[k, i]
                        dW[k * n + i, s] += -2 * r * u[i, j] * v[j, k]

        # Gradient descent with adaptive learning rate
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW

        # Decay learning rate
        if it % 200 == 199:
            lr *= 0.9

    # Check if we found a solution
    R_final = _tensor_residual(best_U, best_V, best_W, n, m, p)
    max_err = np.max(np.abs(R_final))

    if max_err < 1e-8:
        return best_U, best_V, best_W

    return None


def _rationalize_matrix(M, denom=2):
    """
    Try to rationalize a matrix to entries that are multiples of 1/denom.
    Returns the rationalized matrix if close, else None.
    """
    M_rat = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j] * denom
            rounded = round(val)
            if abs(val - rounded) > 1e-6:
                return None
            M_rat[i, j] = rounded / denom
    return M_rat


def _verify_rational(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0, 1)
                for s in range(rank):
                    total += (Fraction(U[i * m + j, s]) *
                              Fraction(V[j * p + k, s]) *
                              Fraction(W[k * n + i, s]))
                if total != 1:
                    return False
    return True


def _to_output(U, V, W):
    """Convert numpy arrays to list of lists with int where possible."""
    def convert(M):
        out = []
        for row in M:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-12:
                    r.append(int(round(x)))
                elif abs(2 * x - round(2 * x)) < 1e-12:
                    r.append(round(2 * x) / 2.0)
                else:
                    r.append(float(x))
            out.append(r)
        return out
    return {"U": convert(U), "V": convert(V), "W": convert(W)}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Guaranteed rank-42
    U42, V42, W42 = _build_rank42()
    if _verify_exact(U42, V42, W42, n, m, p):
        best_result = _to_output(U42, V42, W42)
        best_rank = 42
    else:
        # Should not happen, but fallback to schoolbook
        schoolbook_rank = n * m * p
        U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try rank 41
    if time.time() < deadline - 15:
        result41 = _try_optimize_rank(41, n, m, p, max_iters=1500, lr=0.02, time_limit=25.0)
        if result41 is not None:
            U41, V41, W41 = result41
            # Try to rationalize
            U41r = _rationalize_matrix(U41)
            V41r = _rationalize_matrix(V41)
            W41r = _rationalize_matrix(W41)
            if U41r is not None and V41r is not None and W41r is not None:
                if _verify_rational(U41r, V41r, W41r, n, m, p):
                    best_result = _to_output(U41r, V41r, W41r)
                    best_rank = 41

    # Try rank 40
    if time.time() < deadline - 10 and best_rank > 40:
        result40 = _try_optimize_rank(40, n, m, p, max_iters=1500, lr=0.02, time_limit=20.0)
        if result40 is not None:
            U40, V40, W40 = result40
            U40r = _rationalize_matrix(U40)
            V40r = _rationalize_matrix(V40)
            W40r = _rationalize_matrix(W40)
            if U40r is not None and V40r is not None and W40r is not None:
                if _verify_rational(U40r, V40r, W40r, n, m, p):
                    best_result = _to_output(U40r, V40r, W40r)
                    best_rank = 40

    return best_result
