"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random

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


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    # schoolbook rank
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    # Tensor T[i,j,k] = 1 if (i,j,k) is a valid triple; the multiplication tensor
    # has T[i,j,k]=1 for all i,j,k (it's the "all-ones" tensor in this index layout).
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We try to find a lower-rank exact decomposition over rationals.
    #
    # Strategy: start from schoolbook and attempt column-merging / linear
    # dependence removal to reduce rank. The multiplication tensor for small
    # sizes may admit a rank reduction via Strassen-like tricks.
    #
    # We perform a randomized search: build candidate factor matrices with
    # small integer entries and verify exactness.

    # First, try a structured low-rank ansatz based on grouping.
    # For 3x5x6, try rank values below schoolbook.
    # Known: the tensor rank of the 3x3x3 multiplication tensor is 7 (Strassen).
    # For 3x5x6 we attempt a search.

    # Build the full tensor T as a 3D array: T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p), dtype=object)

    def check(U, V, W):
        # Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        r = U.shape[1]
        # Build the decomposition tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s = s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != T[i, j, k]:
                        return False
        return True

    # Randomized search over small integer entries for low ranks.
    # We focus on ranks that are plausible: between a lower bound and schoolbook.
    # Lower bound: max(n*m, m*p, p*n) = max(15,30,18)=30, and also n*m*p is upper.
    # Try a few candidate ranks below schoolbook.

    random.seed(42)
    for target_rank in range(max(n * m, m * p, p * n), best_rank):
        if time.time() > deadline:
            break
        # Try a few random initializations
        attempts = 0
        max_attempts = 3
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            U = np.zeros((n * m, target_rank), dtype=object)
            V = np.zeros((m * p, target_rank), dtype=object)
            W = np.zeros((p * n, target_rank), dtype=object)
            # Initialize with small integers
            for idx in range(n * m):
                for c in range(target_rank):
                    U[idx, c] = random.randint(-2, 2)
            for idx in range(m * p):
                for c in range(target_rank):
                    V[idx, c] = random.randint(-2, 2)
            for idx in range(p * n):
                for c in range(target_rank):
                    W[idx, c] = random.randint(-2, 2)
            if check(U, V, W):
                best = (U, V, W)
                best_rank = target_rank
                break

    # If no improvement, return schoolbook.
    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
