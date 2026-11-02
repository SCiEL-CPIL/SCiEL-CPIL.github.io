"""Tensor decomposition for 3x4 * 4x7 matrix multiplication.

Strategy:
- Start from the schoolbook (Strassen-like base) rank-84 decomposition.
- Use a greedy low-rank / structured search to try to reduce rank.
- For each candidate rank r < current_best, attempt to find integer/half-integer
  factor matrices U (12 x r), V (28 x r), W (21 x r) such that
    T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  where T is the matrix-multiplication tensor.
- We use a randomized local-search / hill-climbing approach with a deadline guard.
- The exact verification is done with integer arithmetic (multiply by 2 to handle halves).
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 84


def _build_tensor():
    """Build the full 3D tensor T[i,j,k] = 1 if (i,j,k) corresponds to a multiplication term."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _schoolbook():
    """Return the schoolbook decomposition (rank 84)."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, T):
    """Check if U @ diag(V) @ W^T reproduces T exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    We need: for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j,k]
    """
    r = U.shape[1]
    # For each (i,j,k), compute the dot product
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                val = np.dot(u_row * v_val, w_val)
                if val != T[i, j, k]:
                    return False
    return True


def _to_half_ints(mat):
    """Ensure all entries are integers or half-integers (multiples of 0.5)."""
    mat2 = np.round(mat * 2)
    if not np.allclose(mat2, mat * 2, atol=1e-9):
        return None
    return mat2 / 2.0


def _try_greedy_rank(r, T, U0, V0, W0, deadline):
    """Try to find a rank-r decomposition starting from a perturbed schoolbook.
    Uses random local search with hill climbing on the error.
    """
    start = time.time()
    best_U = U0[:, :r].copy() if r <= U0.shape[1] else None
    if best_U is None:
        return None, float('inf')

    best_V = V0[:, :r].copy()
    best_W = W0[:, :r].copy()

    # Compute initial error
    best_err = _compute_error(best_U, best_V, best_W, T)

    if best_err == 0:
        return best_U, best_V, best_W, best_err

    # Local search
    iterations = 0
    max_iter = 50000
    step_size = 1.0

    while iterations < max_iter and time.time() < deadline:
        iterations += 1
        # Pick a random matrix and entry to perturb
        mat_idx = random.choice([0, 1, 2])
        row = random.randint(0, (mat_idx == 0 and n*m-1) or (mat_idx == 1 and m*p-1) or (p*n-1))
        col = random.randint(0, r - 1)

        # Try small perturbations
        for delta in [-step_size, -0.5, 0.5, step_size]:
            if mat_idx == 0:
                old = best_U[row, col]
                best_U[row, col] = old + delta
            elif mat_idx == 1:
                old = best_V[row, col]
                best_V[row, col] = old + delta
            else:
                old = best_W[row, col]
                best_W[row, col] = old + delta

            err = _compute_error(best_U, best_V, best_W, T)
            if err < best_err:
                best_err = err
                if best_err == 0:
                    break
            else:
                # Revert
                if mat_idx == 0:
                    best_U[row, col] = old
                elif mat_idx == 1:
                    best_V[row, col] = old
                else:
                    best_W[row, col] = old

            if best_err == 0:
                break

        if best_err == 0:
            break

        # Occasionally shrink step size
        if iterations % 5000 == 0:
            step_size *= 0.8

    return best_U, best_V, best_W, best_err


def _compute_error(U, V, W, T):
    """Compute the sum of squared errors (using integer-scaled values)."""
    err = 0.0
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                val = np.dot(u_row * V[j * p + k, :], W[k * n + i, :])
                err += (val - T[i, j, k]) ** 2
    return err


def construct():
    deadline = time.time() + 120  # 120 second deadline
    T = _build_tensor()
    U0, V0, W0 = _schoolbook()

    best_U, best_V, best_W = U0, V0, W0
    best_rank = RANK_SCHOOLBOOK

    # Try ranks from best_rank-1 downward, but with a time budget per rank
    for r in range(RANK_SCHOOLBOOK - 1, 0, -1):
        if time.time() > deadline - 10:
            break  # Leave 10 seconds for finalization

        # Give each rank attempt a sub-budget
        rank_budget = min(20, max(2, int((deadline - time.time()) / (RANK_SCHOOLBOOK - r))))
        rank_deadline = time.time() + rank_budget

        U_r, V_r, W_r, err = _try_greedy_rank(r, T, U0, V0, W0, rank_deadline)
        if err == 0:
            # Found a valid rank-r decomposition
            # Convert to half-integer form
            U_r = _to_half_ints(U_r)
            V_r = _to_half_ints(V_r)
            W_r = _to_half_ints(W_r)
            if U_r is not None and V_r is not None and W_r is not None:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
            else:
                # Entries aren't half-integers; try to round
                U_r2 = np.round(U_r * 2) / 2.0
                V_r2 = np.round(V_r * 2) / 2.0
                W_r2 = np.round(W_r * 2) / 2.0
                if _verify(U_r2, V_r2, W_r2, T):
                    best_U, best_V, best_W = U_r2, V_r2, W_r2
                    best_rank = r
                    break

    # Final verification of the best result
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = RANK_SCHOOLBOOK

    # Convert to list of lists for the required interface
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
