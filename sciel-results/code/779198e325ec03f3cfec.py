"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _schoolbook_solution(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    For each (i,j,k), the tensor entry must equal
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    Uses Fraction arithmetic for exactness.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            v_idx = j * p + k
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True


def _solve_w_for_pattern(U, V, n, m, p, target_rank):
    """
    Given fixed U, V (with a chosen sparse pattern and coefficients), solve for W
    such that for each (k,i):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1   for all j in 0..m-1
    This is a linear system in W[k*n+i, :].
    Returns W as a list of lists of Fractions, or None if unsolvable.
    """
    W = [[Fraction(0)] * target_rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            # Build the linear system: for each j, sum_c A[j, c] * W[w_idx, c] = 1
            # where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            A = [[Fraction(0)] * target_rank for _ in range(m)]
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                for c in range(target_rank):
                    A[j][c] = Fraction(U[u_idx][c]) * Fraction(V[v_idx][c])
            b = [Fraction(1)] * m
            # Solve A * W[w_idx,:] = b
            sol = _solve_linear_system(A, b)
            if sol is None:
                return None
            W[w_idx] = sol
    return W


def _solve_linear_system(A, b):
    """
    Solve A x = b over the rationals using exact Gaussian elimination.
    A is m x n (m equations, n unknowns), b is m.
    Returns x (list of Fractions of length n) or None if no exact solution.
    Uses fraction-free / exact elimination with back-substitution.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Forward elimination
    rank = 0
    pivot_cols = []
    for col in range(n):
        # Find pivot
        pivot_row = None
        for r in range(rank, m):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivot_cols.append(col)
        # Eliminate below
        pivot_val = M[rank][col]
        for r in range(rank + 1, m):
            if M[r][col] != 0:
                factor = M[r][col] / pivot_val
                for cc in range(col, n + 1):
                    M[r][cc] = M[r][cc] - factor * M[rank][cc]
        rank += 1
        if rank == m:
            break
    # Check consistency
    for r in range(rank, m):
        if M[r][n] != 0:
            return None
    # Back substitution
    x = [Fraction(0)] * n
    for r in reversed(range(rank)):
        col = pivot_cols[r]
        s = M[r][n]
        for cc in range(col + 1, n):
            s -= M[r][cc] * x[cc]
        x[col] = s / M[r][col]
    return x


def _try_low_rank_search(n, m, p, time_limit=100):
    """
    Search for a lower-rank exact decomposition.
    Strategy: fix a sparse support pattern for U and V, then solve for W exactly
    via a linear system. This bypasses the combinatorial search over W.
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p

    min_possible_rank = max(n * m, m * p, p * n)

    for target_rank in range(min_possible_rank, best_rank):
        if time.time() - start_time > time_limit - 15:
            break

        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10:
                break

            # Build a sparse pattern for U and V.
            # U has n*m rows, V has m*p rows, W has p*n rows.
            # For each (k,i) W row, we need the m equations to be solvable.
            # The m equations for W[k*n+i] involve U[i*m+j, c] * V[j*p+k, c] for j=0..m-1.
            # We need these m rows of the (m x target_rank) matrix to be linearly independent
            # and span enough to solve.
            #
            # Approach: assign each (i,j) U row and each (j,k) V row a set of "active" columns.
            # Use a structured pattern: for each (i,j), pick a random subset of columns for U.
            # For each (j,k), pick a random subset for V.
            # The product U[i*m+j,c]*V[j*p+k,c] is nonzero only where both are active.
            #
            # To ensure the m equations for each (k,i) are solvable, we want the m vectors
            # (over c) [U[i*m+j,c]*V[j*p+k,c]] to be linearly independent.
            #
            # Let's use a pattern where each U row and V row has a fixed number of nonzeros.

            # Random sparse pattern: each U row has ~target_rank/m nonzeros, similarly V.
            # Assign random coefficients.
            U = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V = [[Fraction(0)] * target_rank for _ in range(m * p)]

            # For each (i,j), pick a random subset of columns and assign random coefficients.
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Choose a random subset of columns
                    num_nonzero = max(1, target_rank // m)
                    cols = random.sample(range(target_rank), min(num_nonzero, target_rank))
                    for c in cols:
                        U[u_idx][c] = Fraction(random.randint(1, 3), random.randint(1, 2))

            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    num_nonzero = max(1, target_rank // m)
                    cols = random.sample(range(target_rank), min(num_nonzero, target_rank))
                    for c in cols:
                        V[v_idx][c] = Fraction(random.randint(1, 3), random.randint(1, 2))

            # Solve for W
            W = _solve_w_for_pattern(U, V, n, m, p, target_rank)
            if W is None:
                continue

            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid solution at this rank, stop searching this rank

        if best_solution is not None:
            break

    if best_solution is not None:
        return best_solution
    return None


def construct():
    n, m, p = DIMENSIONS
    # First try the low-rank search
    result = _try_low_rank_search(n, m, p, time_limit=100)
    if result is not None:
        return result
    # Fallback to schoolbook
    return _schoolbook_solution(n, m, p)
# EVOLVE-BLOCK-END
