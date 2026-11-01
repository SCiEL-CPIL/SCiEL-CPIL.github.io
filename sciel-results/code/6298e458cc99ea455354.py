"""Ternary covering code K_3(7,1) via greedy set cover + redundancy removal."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s wall-clock

    # Represent each word as an integer in [0, q**n). Precompute the
    # neighbourhood (ball of radius `radius`) of every word as a list of
    # integer arrays, and the "which codewords cover this word" matrix.
    total = q ** n
    if total > 2000000:
        # Fallback for large spaces: direct product construction.
        return _direct_product(q, n, radius)

    # Build the incidence matrix: M[w, c] = 1 iff word w is within radius of codeword c.
    # We build columns (neighbourhoods) by enumerating each center's ball.
    # For radius 1 the ball size is 1 + n*(q-1).
    ball_size = 1 + n * (q - 1)  # exact for radius 1
    # Build a flat array of all (center, neighbor) pairs.
    # For each center, enumerate all neighbors within Hamming distance <= radius.
    # For radius 1: center itself + single-coordinate changes.
    # Vectorized: for each center c, neighbors are c (itself) and c +/- delta on each coord.

    # Precompute per-coordinate offsets for radius 1.
    # For a center with digits d_0..d_{n-1}, a neighbor at coord i with value v != d_i
    # has integer value c + (v - d_i) * q^i.

    powers = [1]
    for i in range(1, n):
        powers.append(powers[-1] * q)

    # For each center, build its neighborhood as a small list of ints.
    # To vectorize, we build a (total, ball_size) int array.
    centers = np.arange(total, dtype=np.int64)

    # Extract digits of all centers: shape (total, n)
    # digit[i] = (center // q^i) % q
    digits = np.zeros((total, n), dtype=np.int64)
    for i in range(n):
        digits[:, i] = (centers // powers[i]) % q

    # Build neighbourhood flat: for each center, list of neighbor ints.
    # Radius 1: neighbors = [center] + for each coord i, for each v in 0..q-1, v != d_i:
    #   center + (v - d_i) * q^i
    # We'll build a (total, ball_size) array.
    neigh = np.zeros((total, ball_size), dtype=np.int64)
    neigh[:, 0] = centers
    idx = 1
    for i in range(n):
        # For each center, for each v in 0..q-1, v != digits[:,i]:
        # neighbor = center + (v - d_i) * q^i
        # We need to fill ball_size entries.
        # For coord i, there are (q-1) neighbors.
        d = digits[:, i]  # (total,)
        p = powers[i]
        for v in range(q):
            mask = (d != v)
            # For centers where d != v, the neighbor is center + (v - d)*p
            delta = (v - d) * p
            neigh[mask, idx] = centers[mask] + delta[mask]
            idx += 1
    assert idx == ball_size, f"idx {idx} != ball_size {ball_size}"

    # Incidence matrix M: M[w, c] = 1 iff c's neighborhood contains w.
    # Equivalently, M[w, c] = 1 iff w is in neigh[c].
    # Build as a boolean matrix (total, total). For total=2187, that's ~4.8M bools = ~5MB.
    M = np.zeros((total, total), dtype=bool)
    # For each center c, set M[neigh[c], c] = True.
    # Vectorized: use advanced indexing.
    # neigh has shape (total, ball_size). We want M[neigh[c, k], c] = 1 for all c, k.
    row_idx = neigh.reshape(-1)  # (total * ball_size,)
    col_idx = np.repeat(np.arange(total, dtype=np.int64), ball_size)  # (total * ball_size,)
    M[row_idx, col_idx] = True

    # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words.
    # uncovered: boolean array of words not yet covered.
    uncovered = np.ones(total, dtype=bool)
    # count[c] = number of uncovered words covered by codeword c.
    # Initially count[c] = M[:, c].sum()
    count = M.sum(axis=0).astype(np.int64)  # (total,)

    chosen = []
    covered_mask = np.zeros(total, dtype=bool)

    while uncovered.any():
        if time.time() > deadline - 5.0:
            break
        # Pick the codeword with max count among those with count > 0.
        # For ties, pick smallest index for determinism.
        best = int(np.argmax(count))
        if count[best] == 0:
            break
        chosen.append(best)
        # Update: mark covered words.
        col = M[:, best]  # words covered by codeword `best`
        new_covered = col & ~covered_mask
        covered_mask |= col
        # Update count: for each codeword c, subtract the number of newly covered words it also covers.
        # new_covered has shape (total,). For each c, count[c] -= sum(new_covered & M[:, c]).
        # M[:, c] is a column. We need M[new_covered, :].sum(axis=0) but only where new_covered.
        # Vectorized: count -= M[new_covered, :].sum(axis=0)
        if new_covered.any():
            count -= M[new_covered, :].sum(axis=0)
        # Ensure non-negative (safety).
        count = np.maximum(count, 0)

    if not chosen:
        # Fallback: should not happen.
        return _direct_product(q, n, radius)

    # Convert chosen integer codewords back to digit lists.
    codewords = []
    for c in chosen:
        word = []
        for i in range(n):
            word.append((c // powers[i]) % q)
        codewords.append(word)

    # Redundancy removal: try to remove each codeword and re-check coverage.
    # We'll do multiple passes, removing redundant codewords.
    if time.time() < deadline - 3.0:
        codewords = _remove_redundancy(codewords, q, n, radius, M, deadline)

    return codewords


def _remove_redundancy(codewords, q, n, radius, M, deadline):
    """Iteratively remove codewords that are redundant (their covered words are covered by others)."""
    total = q ** n
    powers = [1]
    for i in range(1, n):
        powers.append(powers[-1] * q)

    # Convert to integers for M lookup.
    def to_int(word):
        v = 0
        for i, d in enumerate(word):
            v += d * powers[i]
        return v

    ints = [to_int(w) for w in codewords]
    m = len(ints)

    # coverage_count[w] = number of chosen codewords that cover w.
    # Build from M columns.
    coverage = np.zeros(total, dtype=np.int64)
    for c in ints:
        coverage += M[:, c]

    # A codeword c is redundant if every word it covers is covered by at least one other chosen codeword.
    # i.e., for all w in M[:, c], coverage[w] >= 2.
    # Equivalently, there is no w such that M[w, c] and coverage[w] == 1.

    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        new_ints = []
        for c in ints:
            if time.time() > deadline - 4.0:
                new_ints.append(c)
                continue
            # Check if c is redundant.
            # Words covered by c: M[:, c].
            # c is redundant if for all w with M[w, c], coverage[w] >= 2.
            # i.e., no w with M[w, c] and coverage[w] == 1.
            col = M[:, c]
            # If there exists w with col[w] and coverage[w] == 1, then c is NOT redundant.
            # Check: (col & (coverage == 1)).any()
            if not (col & (coverage == 1)).any():
                # c is redundant: remove it.
                coverage -= col
                changed = True
            else:
                new_ints.append(c)
        ints = new_ints

    # Convert back to digit lists.
    result = []
    for c in ints:
        word = []
        for i in range(n):
            word.append((c // powers[i]) % q)
        result.append(word)
    return result


def _direct_product(q, n, radius):
    """Fallback: direct product of a small perfect/covering code with full space."""
    import itertools
    # For q=3, n=7, radius=1: use tetracode (9 words, length 4) x full space (length 3).
    if q == 3 and n == 7:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    # Generic fallback: all-zero codeword repeated? No, must be distinct.
    # Just return a single codeword (valid only if radius >= n).
    # For safety, return a reasonable set.
    return [[0] * n]
# EVOLVE-BLOCK-END
