"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    # Precompute for speed
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _check_exact_fast(U, V, W, n, m, p):
    """Faster check using numpy where possible, with object dtype for exactness."""
    r = U.shape[1]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Build via einsum-like manual loop
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 1:
                    return False
    return True


def _try_random_decomposition(n, m, p, target_rank, seed, max_iter=200, val_range=(-3, 3)):
    """Try to find a random decomposition with given rank."""
    rng = random.Random(seed)
    U = np.zeros((n * m, target_rank), dtype=object)
    V = np.zeros((m * p, target_rank), dtype=object)
    W = np.zeros((p * n, target_rank), dtype=object)
    for idx in range(n * m):
        for c in range(target_rank):
            U[idx, c] = rng.randint(val_range[0], val_range[1])
    for idx in range(m * p):
        for c in range(target_rank):
            V[idx, c] = rng.randint(val_range[0], val_range[1])
    for idx in range(p * n):
        for c in range(target_rank):
            W[idx, c] = rng.randint(val_range[0], val_range[1])
    return U, V, W


def _structured_strassen_like(n, m, p):
    """
    Try a structured decomposition inspired by Strassen's algorithm.
    For 3x3x3, Strassen achieves rank 7.
    For general sizes, we can use block decomposition.
    """
    # This is a placeholder for a more sophisticated approach
    return None


def _search_rank(n, m, p, start_rank, end_rank, deadline, max_attempts_per_rank=5):
    """Search for a valid decomposition with rank in [start_rank, end_rank]."""
    best = None
    best_rank = end_rank + 1
    for target_rank in range(start_rank, end_rank + 1):
        if time.time() > deadline:
            break
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            U, V, W = _try_random_decomposition(n, m, p, target_rank, seed=42 + target_rank * 100 + attempt)
            if _check_exact(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = target_rank
                break
        if best is not None:
            break
    return best, best_rank


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    n, m, p = DIMENSIONS

    # Schoolbook as fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    # Search for lower rank
    # Lower bound considerations:
    # - The tensor has n*m*p = 90 non-zero entries in the "flat" sense
    # - But the structure allows for lower rank
    # - For 3x3x3, rank is 7 (Strassen)
    # - For 3x5x6, let's search from a reasonable lower bound
    #
    # A simple lower bound: the tensor must have rank at least max(n*m, m*p, p*n) / something
    # Actually, a better lower bound comes from the fact that we need to represent
    # all 90 products. But Strassen shows this can be much less than n*m*p.
    #
    # Let's try ranks from 30 (a rough lower bound) up to 90.
    # Start with a few targeted searches.

    # Try a small range of ranks with random search
    # Focus on ranks that are plausible improvements
    search_start = 30
    search_end = best_rank - 1

    result, found_rank = _search_rank(n, m, p, search_start, search_end, deadline, max_attempts_per_rank=3)
    if result is not None:
        best = result
        best_rank = found_rank

    # If we found an improvement, try to go even lower
    if best_rank < best_rank:
        pass

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
