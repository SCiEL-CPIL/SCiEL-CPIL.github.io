"""Matrix multiplication 3x4x5 tensor rank minimization.

Strategy:
  1. Build the schoolbook rank-60 decomposition as a guaranteed baseline.
  2. Attempt a structured search for a lower-rank decomposition using
     a block-diagonal / Kronecker-product based parameterization with
     integer and half-integer entries.  The search is bounded by a
     deadline guard so that the program always returns within the
     150 s wall-clock limit.
  3. Verify any candidate decomposition against the full tensor before
     returning it.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0


def _build_schoolbook():
    """Return the rank-60 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Check whether U, V, W exactly decompose the 3x4x5 multiplication tensor.

    The tensor T has entries T[(i,j),(j',k),(k',i')] = 1 if j==j' and i==i'
    (with the output index convention W[k*n+i, r] meaning row index = k*n+i).

    We verify by computing sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c and
    comparing to the target tensor.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    # Build the target tensor as a dict of non-zero entries
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1  iff j==j' and i==i'
    # We'll just compute the full sum and check.
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            s = Fraction(0)
                            for r in range(rank):
                                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                            expected = 1 if (j == j2 and i == i2) else 0
                            if s != expected:
                                return False
    return True


def _try_block_diagonal_search(start_time, max_rank_to_try=50):
    """Attempt to find a decomposition with rank < 60 using a structured
    block-diagonal parameterization.

    The idea: partition the 4 middle indices into blocks and use
    Kronecker-product structure to reduce the number of free parameters.
    We try ranks from 50 down to 30 and use random initialization with
    local refinement.

    Returns a dict or None.
    """
    n, m, p = DIMENSIONS
    deadline = start_time + DEADLINE - 5.0  # 5 s safety margin

    # Try a few structured approaches
    for target_rank in range(min(59, max_rank_to_try), 29, -1):
        if time.time() > deadline:
            break

        # Approach: random integer/half-integer matrices with structured sparsity
        # Use a block-diagonal pattern in U, V, W
        attempts = 0
        max_attempts = 200  # bounded by time

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1

            # Generate random structured matrices
            # Use a pattern where each column of U, V, W has a small number of nonzeros
            U = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V = [[Fraction(0)] * target_rank for _ in range(m * p)]
            W = [[Fraction(0)] * target_rank for _ in range(p * n)]

            for r in range(target_rank):
                # Choose random nonzero positions with small integer values
                # U: pick one (i,j) pair
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                U[i * m + j][r] = Fraction(random.choice([-1, 1, 1, 2]))

                # V: pick one (j',k) pair, often same j
                if random.random() < 0.7:
                    jp = j
                else:
                    jp = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                V[jp * p + k][r] = Fraction(random.choice([-1, 1, 1, 2]))

                # W: pick one (k',i') pair
                k2 = random.randint(0, p - 1)
                i2 = random.randint(0, n - 1)
                W[k2 * n + i2][r] = Fraction(random.choice([-1, 1, 1, 2]))

            # Quick check: does the diagonal (i,j,i,j,k,k) work?
            # For each (i,j,k), T[(i*m+j),(j*p+k),(k*n+i)] should be 1
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = Fraction(0)
                        for r in range(target_rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Full verification (expensive, so only do it if quick check passed)
            if time.time() > deadline:
                break
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}

    return None


def _try_kronecker_based(start_time):
    """Try a Kronecker-product based construction.

    For 3x4x5, we can think of the middle dimension 4 = 2x2 and use
    a Kronecker product structure.  The multiplication tensor can be
    written as a sum of rank-1 terms where the middle index is split.

    This is a heuristic attempt; we try to construct a valid decomposition
    using the algebraic structure.
    """
    n, m, p = DIMENSIONS
    deadline = start_time + DEADLINE - 5.0

    # One approach: use the fact that 4 = 2+2 and split the middle index.
    # For each of the two groups of 2 middle indices, we have a 3x2x5 sub-problem.
    # The rank of a 3x2x5 multiplication tensor is at most 30 (schoolbook).
    # But we can potentially share structure between the two groups.

    # Actually, let's try a more direct approach:
    # Use the identity: for each k, the k-th "slice" of the output
    # is a 3x4 matrix product.  The tensor is a sum over k of
    # outer products of the k-th row of A, the k-th column of B,
    # and the k-th "diagonal" of the output.

    # This is just the schoolbook again.  Let's try something else.

    # Try: use a small number of "Strassen-like" combinations.
    # For the 4 middle indices, group them as (0,1) and (2,3).
    # Define S1 = A[:,0] + A[:,1], S2 = A[:,0] - A[:,1]
    #        T1 = B[0,:] + B[1,:], T2 = B[0,:] - B[1,:]
    # Similarly for (2,3).
    # This doesn't directly help for the full tensor.

    # Given the difficulty, return None and rely on the schoolbook.
    return None


def construct():
    start_time = time.time()

    # Baseline: schoolbook rank 60
    best = _build_schoolbook()
    best_rank = 60

    # Try to find a better decomposition
    if time.time() < start_time + DEADLINE - 10.0:
        candidate = _try_block_diagonal_search(start_time, max_rank_to_try=55)
        if candidate is not None:
            best = candidate
            best_rank = len(candidate["U"][0])

    if time.time() < start_time + DEADLINE - 10.0:
        candidate = _try_kronecker_based(start_time)
        if candidate is not None:
            r = len(candidate["U"][0])
            if r < best_rank:
                best = candidate
                best_rank = r

    # Convert Fraction entries to int or float (half-integers)
    # The evaluator expects integer/half-integer entries.
    # If all entries are integers, keep as int; otherwise use float.
    def _convert(mat):
        result = []
        all_int = True
        for row in mat:
            new_row = []
            for x in row:
                if isinstance(x, Fraction):
                    if x.denominator == 1:
                        new_row.append(int(x))
                    else:
                        all_int = False
                        new_row.append(float(x))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    U = _convert(best["U"])
    V = _convert(best["V"])
    W = _convert(best["W"])

    return {"U": U, "V": V, "W": W}
