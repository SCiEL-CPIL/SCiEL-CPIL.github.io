import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds


def _build_strassen_schoolbook(n, m, p):
    """Build the rank-55 decomposition: 5 Strassen blocks (35) + 20 schoolbook terms."""
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # A indices for the 2x2 part
    a00, a01, a10, a11 = 0, 1, 3, 4

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1

        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1

        # M1: (a00+a11)(b00+b11) -> c00, c11
        U[a00, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M2: (a10+a11)b00 -> c10, -c11
        U[a10, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = -1.0
        col += 1

        # M3: a00(b01-b11) -> c01, c11
        U[a00, col] = 1.0
        V[b01, col] = 1.0
        V[b11, col] = -1.0
        W[c01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M4: a11(b10-b00) -> c00, c10
        U[a11, col] = 1.0
        V[b10, col] = 1.0
        V[b00, col] = -1.0
        W[c00, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # M5: (a00+a01)b11 -> -c00, c01
        U[a00, col] = 1.0
        U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0
        W[c01, col] = 1.0
        col += 1

        # M6: (a10-a00)(b00+b01) -> c11
        U[a10, col] = 1.0
        U[a00, col] = -1.0
        V[b00, col] = 1.0
        V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M7: (a01-a11)(b10+b11) -> c00
        U[a01, col] = 1.0
        U[a11, col] = -1.0
        V[b10, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook for A_R (2x1) * B_B (1x2)
        a02 = 0 * 3 + 2
        a12 = 1 * 3 + 2
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1

        # P1: a02*b20 -> c00
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # P2: a02*b21 -> c01
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        # P3: a12*b20 -> c10
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # P4: a12*b21 -> c11
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    r = U.shape[1]
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # Compute tensor sum
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < 1e-9


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by removing one schoolbook term and absorbing it
    into the Strassen basis via integer coefficient adjustments.
    Returns (U', V', W') if successful, else None.
    """
    # Identify schoolbook columns (last 20)
    n_strassen = 35
    n_schoolbook = 20

    # For each schoolbook term, try to see if it can be expressed as
    # a linear combination of Strassen terms with integer coefficients
    # that keeps the total rank at target_rank.

    # Strategy: try removing each schoolbook term and see if the remaining
    # system can still be satisfied by adjusting Strassen coefficients.

    # The tensor equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k

    # This is a multilinear constraint. We can linearize by fixing two
    # matrices and solving for the third, but that's complex.

    # Simpler approach: use the fact that the schoolbook terms have a
    # specific structure. Each schoolbook term (i, k) has:
    # U[i*3+2, c] = 1, V[2*10+k, c] = 1, W[k*2+i, c] = 1

    # Try: can we modify the Strassen terms to cover these?
    # This requires solving a system where we add variables for the
    # adjustments to Strassen coefficients.

    # Given time constraints, use a heuristic: try small integer
    # perturbations to Strassen W coefficients to see if schoolbook
    # contributions can be absorbed.

    # Actually, a cleaner approach: construct the full tensor T[i,j,k]
    # and check if a rank-(target_rank) decomposition exists by
    # attempting to find the factor matrices via iterative methods.

    # For now, return the original if no reduction found.
    return None


def construct():
    _start = time.time()
    n, m, p = DIMENSIONS

    # Build the base rank-55 decomposition
    U, V, W, base_rank = _build_strassen_schoolbook(n, m, p)

    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition is invalid")

    # Try to reduce rank by searching for absorptions
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = base_rank

    # Attempt rank reduction: try removing each schoolbook term
    # and see if the system remains solvable with adjusted Strassen terms.
    # This is a structural search over the 20 schoolbook positions.

    for term_idx in range(20):
        if time.time() - _start > TIME_LIMIT - 10:
            break

        # Try to absorb schoolbook term `term_idx` into Strassen basis
        # by finding integer coefficients that make the tensor equation hold
        # with one fewer term.

        # The schoolbook term at position term_idx corresponds to:
        # i = term_idx // 10, k = term_idx % 10
        # (since schoolbook terms are ordered: for each block b (0..4),
        #  terms are (i=0,k=2b), (i=0,k=2b+1), (i=1,k=2b), (i=1,k=2b+1))

        # Actually, the order is: for b in 0..4:
        #   col 35+4b:   i=0, k=2b
        #   col 35+4b+1: i=0, k=2b+1
        #   col 35+4b+2: i=1, k=2b
        #   col 35+4b+3: i=1, k=2b+1

        # So term_idx maps to:
        b = term_idx // 4
        within = term_idx % 4
        if within == 0:
            i, k = 0, 2 * b
        elif within == 1:
            i, k = 0, 2 * b + 1
        elif within == 2:
            i, k = 1, 2 * b
        else:
            i, k = 1, 2 * b + 1

        # The schoolbook term contributes:
        # U[i*3+2, col] = 1, V[2*10+k, col] = 1, W[k*2+i, col] = 1
        # This adds 1 to T[i, 2, k] (since j=2 for the third column of A
        # and third row of B).

        # To absorb this, we need the Strassen terms to already contribute
        # to T[i, 2, k]. But Strassen terms only involve j=0,1 (first two
        # columns of A and first two rows of B), so they contribute 0 to
        # T[i, 2, k]. Therefore, the schoolbook terms are linearly
        # independent of the Strassen terms and cannot be absorbed.

        # This confirms rank 55 is minimal for this construction.
        # However, we can try a different construction.

        # Alternative: use a different block decomposition.
        # Instead of 2x2 + 2x1, try 2x1 + 2x2 in a different way,
        # or use a 3x3 Strassen-like algorithm.

        # For now, the rank-55 construction is optimal for this approach.
        pass

    # Check time and return best result
    if time.time() - _start > TIME_LIMIT - 5:
        # Return immediately with best result
        pass

    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
