"""
Find a rank-minimal exact tensor decomposition for 2x3 by 3x8 matrix multiplication.

The tensor has 48 entries (2*3*8). We search for the smallest rank r such that there
exist matrices U (6 x r), V (24 x r), W (16 x r) with integer/half-integer entries
satisfying the decomposition constraints.

We use a two-stage approach:
1. Quick check for small ranks (r < 20) using random initialization and gradient descent
   with periodic rounding to integer/half-integer grid.
2. If not found, use a structured search for rank up to 35.
3. Fallback to schoolbook (rank 48).

The key insight: for small n, m, p the rank is often less than n*m*p. For 2x3x8,
we try to find the minimal rank.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    # Build the tensor from the factorization
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_grid(x, tol=1e-6):
    """Round each element to nearest integer or half-integer."""
    x = np.array(x, dtype=float)
    # Nearest integer or half-integer: round(x*2)/2
    return np.round(x * 2.0) / 2.0


def _gradient_step(U, V, W, lr, n, m, p):
    """
    One gradient descent step to minimize sum of squared errors.
    T_target[i,j,k] = 1 for all i,j,k.
    T_factored[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    """
    rank = U.shape[1]
    # Compute errors
    errors = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                errors[(i, j, k)] = val - 1.0

    # Gradients
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)

    for (i, j, k), err in errors.items():
        for r in range(rank):
            u_val = U[i * m + j, r]
            v_val = V[j * p + k, r]
            w_val = W[k * n + i, r]
            dU[i * m + j, r] += 2 * err * v_val * w_val
            dV[j * p + k, r] += 2 * err * u_val * w_val
            dW[k * n + i, r] += 2 * err * u_val * v_val

    U -= lr * dU
    V -= lr * dV
    W -= lr * dW
    return U, V, W


def _try_rank(rank, n, m, p, max_iters=5000, seed=0, deadline=None):
    """
    Try to find a valid decomposition of given rank using gradient descent
    with periodic rounding. Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5

    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    loss += (val - 1.0) ** 2

        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_loss < 1e-10:
            # Check if we can round
            U_r = _round_to_grid(best_U)
            V_r = _round_to_grid(best_V)
            W_r = _round_to_grid(best_W)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r

        # Gradient step with adaptive learning rate
        lr = 0.01 / (1 + it * 0.01)
        U, V, W = _gradient_step(U, V, W, lr, n, m, p)

        # Periodic rounding attempt
        if it % 100 == 0:
            U_r = _round_to_grid(U)
            V_r = _round_to_grid(V)
            W_r = _round_to_grid(W)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r

    # Final check
    U_r = _round_to_grid(best_U)
    V_r = _round_to_grid(best_V)
    W_r = _round_to_grid(best_W)
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
        return U_r, V_r, W_r
    return None


def _structured_search(rank, n, m, p, deadline=None):
    """
    Structured search: exploit the fact that for small matrices,
    we can try binary entries first.
    """
    if deadline is not None and time.time() > deadline:
        return None

    rng = random.Random(42)

    # Try random binary matrices
    for attempt in range(200):
        if deadline is not None and time.time() > deadline:
            break

        U = rng.choice([-1, 0, 1], size=(n * m, rank))
        V = rng.choice([-1, 0, 1], size=(m * p, rank))
        W = rng.choice([-1, 0, 1], size=(p * n, rank))

        # Scale: each product should sum to 1
        # Normalize: divide by sqrt of column norms roughly
        # Actually just check
        if _verify(U.astype(float), V.astype(float), W.astype(float), n, m, p, tol=1e-8):
            return U.astype(float), V.astype(float), W.astype(float)

    return None


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Known: rank of 2x3x8 tensor is likely around 24-40
    # Try from 24 upward

    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # First try structured/binary search for small ranks
    for rank in range(24, 48):
        if time.time() > deadline - 20:
            break

        result = _structured_search(rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {
                "U": [[int(x) for x in row] for row in U],
                "V": [[int(x) for x in row] for row in V],
                "W": [[int(x) for x in row] for row in W]
            }
            return best_result

    # Try gradient-based search for ranks 24 to 47
    for rank in range(24, min(48, best_rank)):
        if time.time() > deadline - 10:
            break

        for seed in range(3):
            if time.time() > deadline - 5:
                break

            result = _try_rank(rank, n, m, p, max_iters=3000, seed=seed, deadline=deadline)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                return best_result

    return best_result

# EVOLVE-BLOCK-END
