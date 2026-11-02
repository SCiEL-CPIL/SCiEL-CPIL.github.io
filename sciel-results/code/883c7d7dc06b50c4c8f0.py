"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a lower rank exact decomposition
    # Known minimal rank for (2,3,3) is 14
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline:
            break

        time_budget = 30 if target_rank == 14 else 15
        rank_deadline = time.time() + time_budget

        result = _try_rank_als(n, m, p, target_rank, rank_deadline)
        if result is not None:
            return result

    return {"U": U_school, "V": V_school, "W": W_school}


def _try_rank_als(n, m, p, rank, deadline):
    """
    Try to find an exact rank decomposition using ALS and integer snapping.
    """
    R = rank
    T = np.ones((n, m, p))

    max_restarts = 30
    for seed in range(max_restarts):
        if time.time() > deadline:
            break

        # Structured initialization combined with random
        U, V, W = _structured_init(n, m, p, R, seed)

        # ALS Iterations
        converged = False
        for it in range(200):
            if time.time() > deadline:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    ATA = A.T @ A
                    ATb = A.T @ b
                    ATA += 1e-12 * np.eye(R)
                    try:
                        U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        U[i * m + j, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    BTA = B.T @ B
                    BTb = B.T @ b
                    BTA += 1e-12 * np.eye(R)
                    try:
                        V[j * p + k, :] = np.linalg.solve(BTA, BTb)
                    except np.linalg.LinAlgError:
                        V[j * p + k, :] = np.linalg.lstsq(BTA, BTb, rcond=None)[0]

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    CTA = C.T @ C
                    CTb = C.T @ b
                    CTA += 1e-12 * np.eye(R)
                    try:
                        W[k * n + i, :] = np.linalg.solve(CTA, CTb)
                    except np.linalg.LinAlgError:
                        W[k * n + i, :] = np.linalg.lstsq(CTA, CTb, rcond=None)[0]

            # Check residual using einsum
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            max_res = np.max(np.abs(result - T))

            if max_res < 1e-10:
                converged = True
                break

        if converged:
            # Polish pass with tighter tolerance
            U, V, W = _polish(n, m, p, R, U, V, W, deadline)

            # Try to snap to integers first
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)

            if _verify_exact(n, m, p, R, U_int, V_int, W_int):
                return _to_list(U_int, V_int, W_int, n, m, p, R)

            # Try to snap to half-integers (denominator 2)
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0

            if _verify_exact(n, m, p, R, U_half, V_half, W_half):
                return _to_list(U_half, V_half, W_half, n, m, p, R)

            # Try small denominators
            for denom in [3, 4, 5, 6]:
                U_snap = np.round(U * denom) / denom
                V_snap = np.round(V * denom) / denom
                W_snap = np.round(W * denom) / denom
                if _verify_exact(n, m, p, R, U_snap, V_snap, W_snap):
                    return _to_list(U_snap, V_snap, W_snap, n, m, p, R)

    return None


def _structured_init(n, m, p, R, seed):
    """Structured initialization for ALS."""
    rng = np.random.RandomState(seed)

    U = rng.randn(n * m, R) * 0.1
    V = rng.randn(m * p, R) * 0.1
    W = rng.randn(p * n, R) * 0.1

    # Add structured components to help convergence
    for r in range(R):
        if r % 3 == 0:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i + j + k) % R == r:
                            U[i * m + j, r] += 1.0
                            V[j * p + k, r] += 1.0
                            W[k * n + i, r] += 1.0
        elif r % 3 == 1:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i * 3 + j * 2 + k) % R == r:
                            U[i * m + j, r] += 1.0
                            V[j * p + k, r] += 1.0
                            W[k * n + i, r] += 1.0
        else:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i * 2 + j * 3 + k * 2) % R == r:
                            U[i * m + j, r] += 1.0
                            V[j * p + k, r] += 1.0
                            W[k * n + i, r] += 1.0

    return U, V, W


def _polish(n, m, p, R, U, V, W, deadline):
    """High-precision polish pass."""
    T = np.ones((n, m, p))
    for it in range(50):
        if time.time() > deadline:
            break

        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                ATA = A.T @ A
                ATb = A.T @ b
                ATA += 1e-14 * np.eye(R)
                try:
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass

        for j in range(m):
            for k in range(p):
                B = np.zeros((n, R))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                BTA = B.T @ B
                BTb = B.T @ b
                BTA += 1e-14 * np.eye(R)
                try:
                    V[j * p + k, :] = np.linalg.solve(BTA, BTb)
                except np.linalg.LinAlgError:
                    pass

        for k in range(p):
            for i in range(n):
                C = np.zeros((m, R))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                CTA = C.T @ C
                CTb = C.T @ b
                CTA += 1e-14 * np.eye(R)
                try:
                    W[k * n + i, :] = np.linalg.solve(CTA, CTb)
                except np.linalg.LinAlgError:
                    pass

    return U, V, W


def _verify_exact(n, m, p, R, U, V, W):
    """Verify if U, V, W provide an exact decomposition using Fraction arithmetic."""
    def to_frac(val):
        return Fraction(val).limit_denominator(100)

    U_frac = [[to_frac(U[i, t]) for t in range(R)] for i in range(n * m)]
    V_frac = [[to_frac(V[j, t]) for t in range(R)] for j in range(m * p)]
    W_frac = [[to_frac(W[k, t]) for t in range(R)] for k in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for t in range(R):
                    s += U_frac[i * m + j][t] * V_frac[j * p + k][t] * W_frac[k * n + i][t]
                if s != 1:
                    return False
    return True


def _to_list(U, V, W, n, m, p, R):
    """Convert numpy arrays to list of lists of integers/floats."""
    def arr_to_list(arr, rows):
        res = []
        for i in range(rows):
            row = []
            for t in range(R):
                val = arr[i, t]
                if val == int(val):
                    row.append(int(val))
                else:
                    row.append(float(val))
            res.append(row)
        return res

    return {
        "U": arr_to_list(U, n * m),
        "V": arr_to_list(V, m * p),
        "W": arr_to_list(W, p * n)
    }
# EVOLVE-BLOCK-END
