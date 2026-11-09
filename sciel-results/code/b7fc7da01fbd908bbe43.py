"""
Program that attempts to find a rank-minimal exact tensor decomposition
for multiplying a 3×5 matrix by a 5×7 matrix (matrix-multiplication tensor T_{3,5,7}).

Score = rank (lower is better).  The seed / schoolbook rank is 3*5*7 = 105.

Strategy
--------
1.  Start from the schoolbook decomposition (rank 105) as a safe fallback.
2.  Try to reduce the rank by exploiting algebraic structure:
    •  Pairing trick:  if two terms share the same (i,j) in U and the same
      (j,k) in V, their W columns can be merged, saving 1 rank for each pair.
    •  We systematically search for such pairings that preserve exactness
      over the integers (entries are in {−1, 0, 1} in the merged columns).
3.  If no structural reduction is found within the time budget, fall back
    to the schoolbook answer.

The key observation is that for T_{3,5,7} with all-ones entries, a term
(r) contributes U[i,j]·V[j,k]·W[k,i] to coordinate (i,j,k).  If two terms
r1, r2 have U[r1][i,j]=U[r2][i,j] and V[r1][j,k]=V[r2][j,k] for every
(i,j,k) where both are non-zero, then we can replace the two W columns
with a single column W[k,i] = W1[k,i] + W2[k,i], reducing rank by 1,
provided the resulting W entries remain small integers (they will, since
we only add ±1 or 0 values).

We implement a greedy merge search:
  •  Build the full schoolbook factorisation.
  •  Repeatedly look for pairs of columns that can be merged.
  •  Stop when no more merges are possible or the deadline is reached.

In addition we try a small number of randomised "perturbation + repair"
steps: randomly zero out one column, then try to repair by adjusting
entries of other columns (within {−1,0,1}) to restore exactness.

All validation is exact (integer arithmetic via Python int).
"""
import time
import sys
import random
from itertools import product
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]
N, M, P = 3, 5, 7
SEED_RANK = N * M * P  # 105

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schoolbook(n=N, m=M, p=P):
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _validate(U, V, W, n=N, m=M, p=P):
    """
    Validate that U, V, W exactly decompose the matrix-multiplication tensor.
    For all (i, j, k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ==  1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _validate_np(U, V, W, n=N, m=M, p=P):
    """NumPy-accelerated validation.  Returns True iff exact."""
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    rank = Ua.shape[1]
    # Build the 3-way array T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # For each (i,j,k), sum over r
    # Use einsum:  T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return bool(np.all(T == 1))


# ---------------------------------------------------------------------------
# Structural merge search
# ---------------------------------------------------------------------------

def _try_merge(U, V, W, n=N, m=M, p=P):
    """
    Attempt to merge two columns of (U, V, W) into one, reducing rank by 1.
    Returns (new_U, new_V, new_W) if a valid merge is found, else None.

    A merge of columns a and b is valid if, after replacing them with a
    single column where each entry is the sum of the two entries, the
    decomposition remains exact.  Since we only allow entries in {-1, 0, 1},
    the sum must also be in {-1, 0, 1} (i.e., the two entries must be equal
    or opposite, and not both ±1 in the same direction giving ±2).
    """
    rank = len(U[0])
    if rank <= 1:
        return None

    # Precompute for each column the set of (i,j) where U is non-zero,
    # the set of (j,k) where V is non-zero, and the set of (k,i) where W is non-zero.
    U_support = []
    for r in range(rank):
        sup = set()
        for ij in range(n * m):
            if U[ij][r] != 0:
                sup.add(ij)
        U_support.append(sup)

    V_support = []
    for r in range(rank):
        sup = set()
        for jk in range(m * p):
            if V[jk][r] != 0:
                sup.add(jk)
        V_support.append(sup)

    W_support = []
    for r in range(rank):
        sup = set()
        for ki in range(p * n):
            if W[ki][r] != 0:
                sup.add(ki)
        W_support.append(sup)

    # For a merge of columns a and b to be valid:
    #   For every (i,j,k) in the tensor:
    #     U_a[i,j]*V_a[j,k]*W_a[k,i] + U_b[i,j]*V_b[j,k]*W_b[k,i]
    #       == (U_a[i,j]+U_b[i,j]) * (V_a[j,k]+V_b[j,k]) * (W_a[k,i]+W_b[k,i])
    #   AND  the new entries must be in {-1, 0, 1}.
    #
    # A sufficient (and in practice necessary for small ints) condition:
    #   For every (i,j):  U_a[i,j] == U_b[i,j]  OR  one of them is 0
    #   For every (j,k):  V_a[j,k] == V_b[j,k]  OR  one of them is 0
    #   For every (k,i):  W_a[k,i] == W_b[k,i]  OR  one of them is 0
    #   AND  the sum in each coordinate is in {-1, 0, 1}.
    #
    # But this is too restrictive.  Let's do a direct check.
    #
    # Actually, the most reliable approach:  just try the merge and validate.
    # To keep it fast, we first do a cheap necessary-condition filter.

    # Cheap filter:  the merged column must have entries in {-1,0,1}.
    # That means for every row, U[a][row] + U[b][row] must be in {-1,0,1},
    # and similarly for V and W.

    def _mergeable_cols(a, b):
        # Check that sums stay in {-1, 0, 1}
        for ij in range(n * m):
            s = U[ij][a] + U[ij][b]
            if s not in (-1, 0, 1):
                return False
        for jk in range(m * p):
            s = V[jk][a] + V[jk][b]
            if s not in (-1, 0, 1):
                return False
        for ki in range(p * n):
            s = W[ki][a] + W[ki][b]
            if s not in (-1, 0, 1):
                return False
        return True

    # Try all pairs
    for a in range(rank):
        for b in range(a + 1, rank):
            if not _mergeable_cols(a, b):
                continue

            # Build the merged column
            new_U_col = [U[ij][a] + U[ij][b] for ij in range(n * m)]
            new_V_col = [V[jk][a] + V[jk][b] for jk in range(m * p)]
            new_W_col = [W[ki][a] + W[ki][b] for ki in range(p * n)]

            # Remove columns a and b, insert merged column at position a
            new_U = []
            new_V = []
            new_W = []
            for r in range(rank):
                if r == a or r == b:
                    continue
                new_U.append(U[:][r])  # copy
                new_V.append(V[:][r])
                new_W.append(W[:][r])
            # Insert merged column at position a
            new_U.insert(a, new_U_col)
            new_V.insert(a, new_V_col)
            new_W.insert(a, new_W_col)

            # Validate
            if _validate_np(new_U, new_V, new_W):
                return (new_U, new_V, new_W)

    return None


def _greedy_merge_search(U, V, W, n=N, m=M, p=P, deadline=None):
    """
    Repeatedly try to merge columns until no more merges are possible
    or the deadline is reached.
    """
    if deadline is None:
        deadline = time.time() + 130

    improved = True
    while improved and time.time() < deadline:
        improved = False
        result = _try_merge(U, V, W, n, m, p)
        if result is not None:
            U, V, W = result
            improved = True

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Randomised perturbation + repair
# ---------------------------------------------------------------------------

def _random_repair(U, V, W, n=N, m=M, p=P, deadline=None, seed=42):
    """
    Try to reduce rank by randomly zeroing a column and then repairing
    the decomposition by adjusting other columns.
    """
    if deadline is None:
        deadline = time.time() + 130

    rng = random.Random(seed)
    rank = len(U[0])
    best_rank = rank
    best = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}

    for attempt in range(200):
        if time.time() > deadline:
            break
        if rank <= 1:
            break

        # Pick a random column to zero out
        col_to_zero = rng.randint(0, rank - 1)

        # Compute the "deficit" tensor: what the zeroed column was contributing
        deficit = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j][col_to_zero] * V[j * p + k][col_to_zero] * W[k * n + i][col_to_zero]
                    if val != 0:
                        deficit[(i, j, k)] = deficit.get((i, j, k), 0) + val

        if not deficit:
            continue  # column was all zeros, already redundant

        # Remove the column
        new_U = [row[:] for row in U if True]
        new_V = [row[:] for row in V if True]
        new_W = [row[:] for row in W if True]
        del new_U[col_to_zero]
        del new_V[col_to_zero]
        del new_W[col_to_zero]
        new_rank = rank - 1

        # Try to repair: for each remaining column, try flipping one entry
        # in U, V, or W to compensate.  This is a very local search.
        # For simplicity, try a few random single-entry flips.
        repaired = False
        for flip_attempt in range(50):
            if time.time() > deadline:
                break

            # Make a copy
            trial_U = [row[:] for row in new_U]
            trial_V = [row[:] for row in new_V]
            trial_W = [row[:] for row in new_W]

            # Pick a random column and a random row in U, V, or W
            r_idx = rng.randint(0, new_rank - 1)
            which = rng.randint(0, 2)
            if which == 0:
                row_idx = rng.randint(0, n * m - 1)
                trial_U[row_idx][r_idx] += 1
                if trial_U[row_idx][r_idx] not in (-1, 0, 1):
                    trial_U[row_idx][r_idx] = 1
            elif which == 1:
                row_idx = rng.randint(0, m * p - 1)
                trial_V[row_idx][r_idx] += 1
                if trial_V[row_idx][r_idx] not in (-1, 0, 1):
                    trial_V[row_idx][r_idx] = 1
            else:
                row_idx = rng.randint(0, p * n - 1)
                trial_W[row_idx][r_idx] += 1
                if trial_W[row_idx][r_idx] not in (-1, 0, 1):
                    trial_W[row_idx][r_idx] = 1

            if _validate_np(trial_U, trial_V, trial_W, n, m, p):
                best = {"U": trial_U, "V": trial_V, "W": trial_W}
                best_rank = new_rank
                repaired = True
                # Update working copy
                U = trial_U
                V = trial_V
                W = trial_W
                rank = new_rank
                break

            # Also try flipping by -1
            trial_U = [row[:] for row in new_U]
            trial_V = [row[:] for row in new_V]
            trial_W = [row[:] for row in new_W]
            if which == 0:
                trial_U[row_idx][r_idx] -= 1
                if trial_U[row_idx][r_idx] not in (-1, 0, 1):
                    trial_U[row_idx][r_idx] = -1
            elif which == 1:
                trial_V[row_idx][r_idx] -= 1
                if trial_V[row_idx][r_idx] not in (-1, 0, 1):
                    trial_V[row_idx][r_idx] = -1
            else:
                trial_W[row_idx][r_idx] -= 1
                if trial_W[row_idx][r_idx] not in (-1, 0, 1):
                    trial_W[row_idx][r_idx] = -1

            if _validate_np(trial_U, trial_V, trial_W, n, m, p):
                best = {"U": trial_U, "V": trial_V, "W": trial_W}
                best_rank = new_rank
                repaired = True
                U = trial_U
                V = trial_V
                W = trial_W
                rank = new_rank
                break

        if not repaired:
            # Restore original columns for this attempt
            pass  # We already have the original U, V, W

    return best


# ---------------------------------------------------------------------------
# Structured low-rank construction
# ---------------------------------------------------------------------------

def _structured_search(n=N, m=M, p=P, deadline=None, seed=42):
    """
    Search for a low-rank decomposition using a structured parametrisation.

    We use the fact that the matrix multiplication tensor T_{n,m,p} has
    symmetry:  permuting the index sets (with appropriate relabelling)
    maps T to itself (up to reordering).  We exploit this by trying
    decompositions where U, V, W have a regular pattern.

    One approach:  use a "grid" construction.  Partition the index sets
    into blocks and use a Kronecker-product-like structure.

    For 3x5x7, we can try:
      rank = 3*5*7 - k  for k = 1, 2, ..., up to some limit
    and use the merge search to find valid reductions.

    This function tries multiple random seeds for the merge search.
    """
    if deadline is None:
        deadline = time.time() + 130

    rng = random.Random(seed)
    best = _schoolbook(n, m, p)
    best_rank = SEED_RANK

    # Try the greedy merge from schoolbook with different randomised column orderings
    for trial in range(20):
        if time.time() > deadline:
            break

        # Shuffle the column ordering of the schoolbook factorisation
        U = [[0] * SEED_RANK for _ in range(n * m)]
        V = [[0] * SEED_RANK for _ in range(m * p)]
        W = [[0] * SEED_RANK for _ in range(p * n)]
        perm = list(range(SEED_RANK))
        rng.shuffle(perm)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][perm[col]] = 1
                    V[j * p + k][perm[col]] = 1
                    W[k * n + i][perm[col]] = 1
                    col += 1

        result = _greedy_merge_search(U, V, W, n, m, p, deadline=deadline)
        cur_rank = len(result["U"][0])
        if cur_rank < best_rank:
            best_rank = cur_rank
            best = result

    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    """
    Returns the best tensor decomposition found for 3×5×7 matrix multiplication.
    Interface:  {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank),  V: (m*p, rank),  W: (p*n, rank)
    """
    start = time.time()
    deadline = start + 135  # safety margin before the 150 s kill

    # Safe fallback
    best = _schoolbook()
    best_rank = SEED_RANK

    # Phase 1:  Greedy merge from schoolbook
    if time.time() < deadline - 5:
        result = _greedy_merge_search(best["U"], best["V"], best["W"], deadline=deadline - 5)
        cur_rank = len(result["U"][0])
        if cur_rank < best_rank:
            best_rank = cur_rank
            best = result

    # Phase 2:  Structured search with randomised column orderings
    if time.time() < deadline - 5:
        result = _structured_search(deadline=deadline - 5, seed=123)
        cur_rank = len(result["U"][0])
        if cur_rank < best_rank:
            best_rank = cur_rank
            best = result

    # Phase 3:  Randomised perturbation + repair on the best found so far
    if time.time() < deadline - 2:
        result = _random_repair(best["U"], best["V"], best["W"], deadline=deadline - 2, seed=999)
        cur_rank = len(result["U"][0])
        if cur_rank < best_rank:
            best_rank = cur_rank
            best = result

    # Final validation
    if not _validate(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook if something went wrong
        best = _schoolbook()

    return best


# EVOLVE-BLOCK-END
