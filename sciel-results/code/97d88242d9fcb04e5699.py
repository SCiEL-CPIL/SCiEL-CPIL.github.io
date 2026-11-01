"""Greedy set-cover for the ternary covering code K_3(6,1).

Start from the seed's 81 disjoint packing (direct sum of the ternary
tetracode with the full 2-coordinate tail). Because that packing's Hamming-1
balls are disjoint, the optimum is at most 81, so a greedy set-cover can only
match (81) or improve (<=80). We repeatedly add the candidate codeword that
covers the most still-uncovered words, using a NumPy-vectorized coverage mask
over the 3**6 = 729-word space.
"""
import time
import itertools
import numpy as np


def _all_words(q, n):
    """Return a (q**n, n) uint8 array with every word over {0..q-1}^n."""
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)


def _seed_words(q, n):
    """Seed packing: direct sum of the ternary tetracode (perfect code, radius 1)
    with every word on the remaining n-4 coordinates. Gives q**(n-4) * 9 words
    whose Hamming-1 balls are pairwise disjoint, so it is a valid cover and the
    optimum is at most this size."""
    if n < 4:
        return _all_words(q, n)
    tetracode = np.array(
        [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)],
        dtype=np.uint8,
    )
    tails = _all_words(q, n - 4)
    words = np.empty((len(tetracode) * len(tails), n), dtype=np.uint8)
    words[:, :4] = np.repeat(tetracode, len(tails), axis=0)
    words[:, 4:] = np.tile(tails, (len(tetracode), 1))
    return words


def _coverage_mask(words, allw, q, n, radius):
    """Boolean array: True where every word of allw is within `radius` of some
    codeword in words. Vectorized via a digit-weighted Hamming-distance trick."""
    if words.size == 0:
        return np.zeros(len(allw), dtype=bool)
    # digit weights so that sum of weights == Hamming distance (q**n-1 < q**(n+1))
    w = q ** np.arange(n, dtype=np.int64)
    a = (allw.astype(np.int64) * w).sum(axis=1)
    b = (words.astype(np.int64) * w).sum(axis=1)
    D = np.abs(a[:, None] - b[None, :])
    return (D <= radius).any(axis=1)


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin well under the 150 s limit

    allw = _all_words(q, n)
    N = len(allw)
    seed = _seed_words(q, n)
    candidates = np.vstack([seed, allw])

    covered = _coverage_mask(seed, allw, q, n, radius)
    chosen = [list(row) for row in seed]

    # Greedy set-cover: add the candidate covering the most uncovered words.
    while not covered.all():
        if time.time() > deadline:
            break
        uncovered = ~covered
        # number of still-uncovered words each candidate would cover
        gains = _coverage_mask(candidates, allw, q, n, radius) & uncovered
        gains = gains.sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break  # no candidate helps; keep the valid cover we have
        chosen.append(list(candidates[best]))
        covered |= _coverage_mask(candidates[best:best + 1], allw, q, n, radius)

    return chosen
