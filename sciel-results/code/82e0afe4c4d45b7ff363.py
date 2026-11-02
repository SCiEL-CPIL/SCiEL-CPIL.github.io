"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]


def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) -> indices (i,j)
    # V shape: (m*p, rank) -> indices (j,k)
    # W shape: (p*n, rank) -> indices (k,i)
    #
    # We require:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    #
    # The evaluator expects integer/half-integer entries. We will search for
    # exact rational solutions with small denominators and then scale to
    # integers/half-integers if possible.

    deadline = time.time() + 120  # safety margin for 150s wall-clock

    # Fallback: schoolbook decomposition (always valid)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()

    # Try ranks from low to high.
    # Lower bound is at least max(n*m, m*p, p*n) = 9, but tensor rank is likely higher.
    # We'll try a reasonable range.
    for rank in range(10, 25):
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, rank, deadline=deadline)
        if result is not None:
            return result

    return best_result


def _try_rank(n, m, p, rank, max_iters=2000, tol=1e-10, n_restarts=30, deadline=None):
    """Try to find a rank-r decomposition via alternating minimization, then exact recovery."""
    np.random.seed(42)

    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 10:
            break

        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)

        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        success = _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol)
        if success:
            # Try exact rational recovery
            recovered = _recover_exact_solution(n, m, p, rank, U, V, W)
            if recovered is not None:
                U_rec, V_rec, W_rec = recovered
                if _verify_exact(n, m, p, U_rec, V_rec, W_rec):
                    # Convert to float list for return
                    return {
                        "U": U_rec.tolist(),
                        "V": V_rec.tolist(),
                        "W": W_rec.tolist()
                    }

    return None


def _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol):
    """Alternating least squares to satisfy T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""

    for it in range(max_iters):
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    return False

        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    return False

        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    return False

        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        if err < tol:
            return True

    return False


def _recover_exact_solution(n, m, p, rank, U, V, W, max_denom=16):
    """
    Try to recover an exact rational solution from the floating-point solution.
    We assume entries are close to rationals with small denominators.
    """
    # Try to find a common denominator that makes all entries close to integers
    for denom in range(1, max_denom + 1):
        # Scale up
        U_scaled = U * denom
        V_scaled = V * denom
        W_scaled = W * denom

        # Check if entries are close to integers
        U_int = np.round(U_scaled)
        V_int = np.round(V_scaled)
        W_int = np.round(W_scaled)

        # Check closeness
        if (np.max(np.abs(U_int - U_scaled)) < 1e-6 and
                np.max(np.abs(V_int - V_scaled)) < 1e-6 and
                np.max(np.abs(W_int - W_scaled)) < 1e-6):
            # We have a candidate rational solution
            U_rat = U_int / denom
            V_rat = V_int / denom
            W_rat = W_int / denom

            # Verify exactly
            if _verify_exact(n, m, p, U_rat, V_rat, W_rat):
                # Try to scale to integer/half-integer
                # The evaluator expects integer/half-integer entries.
                # If denom is 1, we have integers. If denom is 2, we have half-integers.
                # For other denominators, we need to find a common multiple.
                # Since we want to minimize rank, we should keep the rank as is.
                # But the entries must be integer or half-integer.
                # If denom is not 1 or 2, we can multiply all entries by lcm(denom, 2) / denom
                # to get half-integers, but this changes the rank (scales the factors).
                # Actually, the decomposition is homogeneous: if (U,V,W) is a solution,
                # then (aU, V/a, W) is also a solution for any nonzero a.
                # So we can rescale to make entries integer/half-integer.

                # Find a scaling factor that makes all entries integer or half-integer.
                # We want all entries to be multiples of 0.5.
                # So we need denom * scale to be a multiple of 2.
                # Let scale = 2 / gcd(denom, 2)
                from math import gcd
                g = gcd(denom, 2)
                scale = 2 / g

                U_final = U_rat * scale
                V_final = V_rat / scale
                W_final = W_rat

                # Verify again
                if _verify_exact(n, m, p, U_final, V_final, W_final):
                    # Check that entries are integer or half-integer
                    if _entries_are_half_integer(U_final) and _entries_are_half_integer(V_final) and _entries_are_half_integer(W_final):
                        return U_final, V_final, W_final

    return None


def _entries_are_half_integer(arr, tol=1e-10):
    """Check if all entries are integer or half-integer."""
    scaled = arr * 2
    int_scaled = np.round(scaled)
    return np.max(np.abs(int_scaled - scaled)) < tol


def _verify_exact(n, m, p, U, V, W, tol=1e-10):
    """
    Verify that the decomposition is exact:
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
