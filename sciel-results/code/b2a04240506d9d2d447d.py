"""Matrix multiplication 4,4,7: rank-minimal tensor decomposition search."""
import time
import math
import random
import itertools

import numpy as np
from numpy.linalg import norm

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) arrays for the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _error(U, V, W, n, m, p):
    """Compute the tensor residual norm for the decomposition."""
    # T_ours[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # T_target[i,j,k] = 1
    # Vectorize: build a 3D tensor from outer products.
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T_ours[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # Target is all ones
    residual = T - 1.0
    return float(np.linalg.norm(residual))


def _is_valid(U, V, W, n, m, p, tol=1e-9):
    return _error(U, V, W, n, m, p) < tol


def _randomize(U, V, W, n, m, p, alpha):
    """Perturb the factors with small random noise."""
    U = U + alpha * np.random.randn(*U.shape)
    V = V + alpha * np.random.randn(*V.shape)
    W = W + alpha * np.random.randn(*W.shape)
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, target_rank, tol=1e-9, max_iter=200):
    """
    Try to reduce the rank by projecting out columns that are nearly redundant.
    We attempt to find a low-rank approximation of each factor that still
    satisfies the tensor equation.
    """
    # This is a heuristic: we try to find a subset of columns that still works.
    # For a proper approach, we'd need to solve a system of equations.
    # Here we just return the original if we can't reduce.
    return U, V, W


def _polish(U, V, W, n, m, p, tol=1e-12, max_iter=100):
    """
    High-precision polish: use least-squares to adjust factors to exactly
    satisfy the tensor equation, given a fixed rank.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]

    # We can try to minimize the residual by adjusting W (or any one factor)
    # given fixed U and V. The equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # For fixed U, V, this is linear in W.
    # Build the linear system: for each (i,j,k), sum_c A_{(i,j,k),c} * W[k*n+i,c] = 1
    # But W[k*n+i,c] appears in the equation for (i,j,k) with coefficient U[i*m+j,c]*V[j*p+k,c]
    # So we have a system of n*m*p equations in r*p*n unknowns (W entries).
    # This is underdetermined if r*p*n > n*m*p, i.e., r > m.
    # Let's build the matrix A where row is (i,j,k) and column is (k*n+i, c).
    # A[(i,j,k), (k*n+i, c)] = U[i*m+j,c] * V[j*p+k,c]
    # Then A @ W_flat = ones(n*m*p)
    # W_flat has shape (p*n*r,)
    # Number of equations: n*m*p = 4*4*7 = 112
    # Number of unknowns: p*n*r = 7*4*r = 28*r
    # For r < 4, unknowns < equations, overdetermined. For r >= 4, underdetermined.
    # We want the minimum-norm solution or just a solution.
    # Use lstsq.

    # Build A
    # A has shape (n*m*p, p*n*r)
    A = np.zeros((n * m * p, p * n * r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k  # row index
                for c in range(r):
                    col = (k * n + i) * r + c  # column index in W_flat
                    A[eq, col] = U[i * m + j, c] * V[j * p + k, c]

    # Solve A @ W_flat = ones
    b = np.ones(n * m * p)
    try:
        W_flat, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=tol)
    except np.linalg.LinAlgError:
        return U, V, W

    W_new = W_flat.reshape(p * n, r)
    # Check error
    if _error(U, V, W_new, n, m, p) < 1e-8:
        W = W_new

    # Now try to polish U given fixed V, W
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For fixed V, W, linear in U.
    # A2[(i,j,k), (i*m+j, c)] = V[j*p+k,c] * W[k*n+i,c]
    A2 = np.zeros((n * m * p, n * m * r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for c in range(r):
                    col = (i * m + j) * r + c
                    A2[eq, col] = V[j * p + k, c] * W[k * n + i, c]
    try:
        U_flat, _, _, _ = np.linalg.lstsq(A2, b, rcond=tol)
        U_new = U_flat.reshape(n * m, r)
        if _error(U_new, V, W, n, m, p) < 1e-8:
            U = U_new
    except np.linalg.LinAlgError:
        pass

    # Polish V given fixed U, W
    # A3[(i,j,k), (j*p+k, c)] = U[i*m+j,c] * W[k*n+i,c]
    A3 = np.zeros((n * m * p, m * p * r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for c in range(r):
                    col = (j * p + k) * r + c
                    A3[eq, col] = U[i * m + j, c] * W[k * n + i, c]
    try:
        V_flat, _, _, _ = np.linalg.lstsq(A3, b, rcond=tol)
        V_new = V_flat.reshape(m * p, r)
        if _error(U, V_new, W, n, m, p) < 1e-8:
            V = V_new
    except np.linalg.LinAlgError:
        pass

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]  # 112

    # Strategy: try to find a lower-rank decomposition by:
    # 1. Random initialization with lower rank + polish
    # 2. Pruning columns from schoolbook
    # 3. Iterative refinement

    random.seed(42)
    np.random.seed(42)

    # Try ranks from a reasonable lower bound up to schoolbook rank
    # The tensor rank of matrix multiplication is at most n*m*p and at least max(n*m, m*p, p*n)
    # For 4,4,7: lower bound is max(16, 28, 28) = 28
    # Known bounds: for M_{4,4,7}, the rank is between 28 and 112.
    # Let's try to find something below 112.

    # First, try to prune the schoolbook decomposition
    # The schoolbook has 112 columns. Can we remove some?
    # Each column corresponds to a (i,j,k) triple. The tensor equation requires
    # that for each (i,j,k), the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    # In schoolbook, each (i,j,k) is covered by exactly one column.
    # To reduce rank, we need to combine columns.

    # Approach: random low-rank init + polish
    for target_rank in [28, 30, 32, 35, 40, 45, 50, 60, 70, 80, 90, 100, 112]:
        if time.time() > deadline:
            break

        # Try several random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break

            # Random init
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5

            # Polish
            U, V, W = _polish(U, V, W, n, m, p, tol=1e-10, max_iter=50)

            err = _error(U, V, W, n, m, p)
            if err < 1e-8 and target_rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break  # Found a valid one at this rank, try lower

            # Also try with larger init scale
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            U, V, W = _polish(U, V, W, n, m, p, tol=1e-10)
            err = _error(U, V, W, n, m, p)
            if err < 1e-8 and target_rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break

    # Try to further reduce by column pruning from the best found
    if best_rank > 28:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        r = U.shape[1]
        # Try removing columns one by one and re-polishing
        for col_to_remove in range(r - 1, 27, -1):
            if time.time() > deadline:
                break
            if col_to_remove >= U.shape[1]:
                continue
            # Remove column
            mask = np.ones(U.shape[1], dtype=bool)
            mask[col_to_remove] = False
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            # Polish
            U_p, V_p, W_p = _polish(U_sub, V_sub, W_sub, n, m, p, tol=1e-10)
            err = _error(U_p, V_p, W_p, n, m, p)
            if err < 1e-8:
                best_U, best_V, best_W = U_p.copy(), V_p.copy(), W_p.copy()
                best_rank = U_p.shape[1]
                # Continue trying to remove more
                U, V, W = U_p, V_p, W_p
            else:
                # Can't remove this column, try next
                pass

    # Final check
    final_err = _error(best_U, best_V, best_W, n, m, p)
    if final_err >= 1e-8:
        # Fall back to schoolbook
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = U0.shape[1]

    # Convert to lists of ints/floats
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
