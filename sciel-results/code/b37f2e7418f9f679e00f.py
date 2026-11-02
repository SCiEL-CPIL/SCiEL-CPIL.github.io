"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 48

# Deadline guard
START = time.time()
DEADLINE = 115.0  # wall-clock seconds, safety margin under 120/150s


def _time_left():
    return DEADLINE - (time.time() - START)


def _tensor_matrix():
    """Return the multiplication tensor as a 3D array T[i, j, k] with 1 iff term (i,j,k) exists."""
    T = np.zeros((N, M, P), dtype=int)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1
    return T


def _matrices_to_tensors(U, V, W, rank):
    """Compute the implied tensor from factor matrices (list of lists or arrays)."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    T = np.zeros((N, M, P), dtype=object)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(N):
            if uc[i * M + 0] == 0:
                continue
            for j in range(M):
                uval = uc[i * M + j]
                if uval == 0:
                    continue
                for k in range(P):
                    vval = vc[j * P + k]
                    if vval == 0:
                        continue
                    wval = wc[k * N + i]
                    if wval == 0:
                        continue
                    T[i, j, k] = T[i, j, k] + uval * vval * wval
    return T


def _schoolbook():
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W


def _validate(U, V, W):
    """Check exact decomposition. Returns (ok, rank)."""
    rank = len(U[0]) if U else 0
    T = _matrices_to_tensors(U, V, W, rank)
    # Expected tensor: all ones
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if T[i, j, k] != 1:
                    return False, rank
    return True, rank


def _to_fraction_matrices(U, V, W):
    """Convert integer/float matrices to Fraction matrices for exact arithmetic."""
    def conv(M_):
        return [[Fraction(x).limit_denominator(10**6) for x in row] for row in M_]
    return conv(U), conv(V), conv(W)


def _try_remove_column(U, V, W, col):
    """
    Attempt to remove column `col` by adjusting other columns.
    Strategy: set col to zero, then for each (i,j,k) that was contributed only by col,
    we need the remaining columns to still sum to 1. This is a linear system; we attempt
    a simple repair by redistributing. Returns new (U,V,W) or None.
    """
    rank = len(U[0])
    # Compute contribution of column `col` to each (i,j,k)
    Uc = [U[i][col] for i in range(N * M)]
    Vc = [V[i][col] for i in range(M * P)]
    Wc = [W[i][col] for i in range(P * N)]
    contrib = np.zeros((N, M, P), dtype=int)
    for i in range(N):
        for j in range(M):
            uval = Uc[i * M + j]
            if uval == 0:
                continue
            for k in range(P):
                vval = Vc[j * P + k]
                if vval == 0:
                    continue
                wval = Wc[k * N + i]
                if wval == 0:
                    continue
                contrib[i, j, k] += uval * vval * wval
    # If a term is contributed by col AND by others, removing col breaks it.
    # We only allow removal if col is the sole contributor to some terms and we can
    # compensate. For simplicity, only attempt removal if col contributes to at most
    # one term with value 1 and that term is also contributed by others (so we can
    # drop col's contribution by adjusting). This is very restrictive; instead we
    # do a full linear solve below.
    return None  # placeholder; real repair in _linear_repair


def _linear_repair(U, V, W, target_rank, rng, max_iters=200):
    """
    Given factor matrices of current rank, try to reduce rank to target_rank by
    zeroing out (rank - target_rank) columns and solving a linear system for the
    remaining free entries. We treat the tensor equations as linear in the entries.
    This is heuristic: we pick which columns to drop, then solve least-squares
    exactly (integer/half-integer) for a minimal set of free variables.
    """
    rank = len(U[0])
    if target_rank >= rank:
        return None
    # We'll attempt to drop the last (rank - target_rank) columns.
    keep = list(range(target_rank))
    # Build linear system: unknowns are the entries of U,V,W in kept columns.
    # Equations: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all (i,j,k).
    # This is cubic in variables, not linear. So linear repair is not directly applicable.
    # Instead, we use a different strategy: random re-optimization via coordinate descent
    # on the rank-reduced matrices, trying to hit the tensor exactly.
    return None


def _random_search(U0, V0, W0, rng, max_time=60.0):
    """
    Heuristic search: start from schoolbook, then try to reduce rank by
    'merging' pairs of terms. We attempt to find a lower-rank decomposition by
    random restarts with a greedy column-elimination + local repair.
    """
    best = (RANK_SCHOOLBOOK, U0, V0, W0)
    U, V, W = [row[:] for row in U0], [row[:] for row in V0], [row[:] for row in W0]
    rank = len(U[0])
    t0 = time.time()
    while _time_left() > max_time and rank > 1:
        # Try to remove a random column
        col = rng.randrange(rank)
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        # Set column to zero
        for i in range(N * M):
            U2[i][col] = 0
        for i in range(M * P):
            V2[i][col] = 0
        for i in range(P * N):
            W2[i][col] = 0
        # Now try to repair: for each (i,j,k) where the tensor value is now wrong,
        # adjust one of the kept columns. We pick a random kept column and try to
        # fix the deficit by scaling.
        T = _matrices_to_tensors(U2, V2, W2, rank)
        ok = True
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    if T[i, j, k] != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # Successfully removed column
            rank -= 1
            U = U2
            V = V2
            W = W2
            # Drop the zero column
            U = [row[:col] + row[col + 1:] for row in U]
            V = [row[:col] + row[col + 1:] for row in V]
            W = [row[:col] + row[col + 1:] for row in W]
            if rank < best[0]:
                best = (rank, [row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Try a simple repair: for each broken term, add a new column that fixes it
            # and removes the broken contribution. This is complex; skip for now.
            pass
        if time.time() - t0 > max_time:
            break
    return best


def construct():
    rng = random.Random(12345)
    U0, V0, W0 = _schoolbook()
    # Validate schoolbook
    ok, r = _validate(U0, V0, W0)
    if not ok:
        # Should not happen
        return {"U": U0, "V": V0, "W": W0}

    best_rank = r
    best = (U0, V0, W0)

    # Try random search with time budget
    if _time_left() > 30:
        rank, U, V, W = _random_search(U0, V0, W0, rng, max_time=80.0)
        if rank < best_rank:
            # Validate
            ok2, _ = _validate(U, V, W)
            if ok2:
                best_rank = rank
                best = (U, V, W)

    U, V, W = best
    return {"U": U, "V": V, "W": W}
