"""Fresh design: tetracode seed + redundant-codeword removal + greedy repair.
Optimized for the canonical ternary q=3,n=6,r=1 case (the task's target),
with a fast NumPy ball-coverage representation and an explicit deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import itertools

import numpy as np

_DEADLINE = None


def _now():
    return time.perf_counter()


def _ball_indices(q, n, radius):
    """For each position and each symbol, the list of (symbol, offsets) that are
    within Hamming distance 1. Used to build the ball of a word as a flat index set."""
    # For a single position i, a word is within distance 1 if we change exactly one
    # of its n positions to a different symbol (radius 1). Precompute the flat index
    # offsets for each (position, new_symbol) relative to a base word.
    # A word w has index sum(w[k] * q**(n-1-k)). Changing position i from old to new
    # adds (new - old) * q**(n-1-i).
    powers = [q ** (n - 1 - i) for i in range(n)]
    # ball_deltas[i][old][new] -> delta
    # We'll just compute on the fly per word; but precompute the per-position symbol
    # multiplier list.
    return powers


def _word_to_idx(word, q, n):
    idx = 0
    for k, v in enumerate(word):
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for k in range(n - 1, -1, -1):
        word[k] = idx % q
        idx //= q
    return word


def _ball_flat_indices(word, q, n, powers):
    """Return the flat index of the word itself plus all words at Hamming distance 1."""
    base = 0
    for k, v in enumerate(word):
        base = base * q + v
    out = [base]
    for i in range(n):
        old = word[i]
        p = powers[i]
        for new in range(q):
            if new != old:
                out.append(base + (new - old) * p)
    return out


def _build_coverage(codewords, q, n, radius):
    """Return a boolean array of length q**n marking covered words, plus a count."""
    total = q ** n
    covered = np.zeros(total, dtype=np.bool_)
    powers = [q ** (n - 1 - i) for i in range(n)]
    for w in codewords:
        idxs = _ball_flat_indices(w, q, n, powers)
        covered[np.array(idxs, dtype=np.int64)] = True
    return covered


def _coverage_count(covered):
    return int(np.count_nonzero(covered))


def _is_covered(covered, word, q, n, powers):
    idxs = _ball_flat_indices(word, q, n, powers)
    return bool(np.all(covered[np.array(idxs, dtype=np.int64)]))


def _greedy_repair(codewords, covered, q, n, radius, powers, deadline):
    """Add codewords to cover uncovered words until full coverage or deadline."""
    total = q ** n
    uncovered_mask = ~covered
    while np.any(uncovered_mask) and _now() < deadline:
        # Find an uncovered word to target.
        target_idx = int(np.argmax(uncovered_mask.astype(np.int8)))
        if not uncovered_mask[target_idx]:
            break
        target_word = _idx_to_word(target_idx, q, n)
        # Greedily pick the candidate word (among all words at distance <=1 of target)
        # whose ball covers the most currently-uncovered words.
        # Candidate set = the ball of target (words within distance 1 of target).
        cand_idxs = _ball_flat_indices(target_word, q, n, powers)
        cand_idxs = np.array(cand_idxs, dtype=np.int64)
        best_cand = None
        best_gain = -1
        # For each candidate, compute how many of its ball positions are uncovered.
        # To be fast, compute ball of candidate and intersect with uncovered_mask.
        for ci in cand_idxs:
            cw = _idx_to_word(int(ci), q, n)
            ball = np.array(_ball_flat_indices(cw, q, n, powers), dtype=np.int64)
            gain = int(np.count_nonzero(uncovered_mask[ball]))
            if gain > best_gain:
                best_gain = gain
                best_cand = cw
        if best_cand is None or best_gain <= 0:
            # Fallback: add the target word itself.
            best_cand = target_word
        # Commit it.
        codewords.append(best_cand)
        ball = np.array(_ball_flat_indices(best_cand, q, n, powers), dtype=np.int64)
        covered[ball] = True
        uncovered_mask = ~covered
    return codewords, covered


def _remove_redundant(codewords, covered, q, n, radius, powers, deadline):
    """Iteratively remove a codeword if the rest still cover everything.
    Returns (codewords, covered)."""
    changed = True
    while changed and _now() < deadline:
        changed = False
        m = len(codewords)
        for i in range(m):
            if _now() > deadline:
                return codewords, covered
            # Temporarily remove codeword i.
            remaining = codewords[:i] + codewords[i + 1:]
            cov2 = np.zeros(q ** n, dtype=np.bool_)
            for w in remaining:
                ball = np.array(_ball_flat_indices(w, q, n, powers), dtype=np.int64)
                cov2[ball] = True
            if np.all(cov2):
                codewords = remaining
                covered = cov2
                changed = True
                break
    return codewords, covered


def construct(q, n, radius):
    global _DEADLINE
    start = _now()
    # Safety margin: evaluator kills at 150s; leave headroom.
    deadline = start + 120.0
    _DEADLINE = deadline

    # Canonical fast path for the task's target: q=3, n=6, radius=1.
    if q == 3 and n == 6 and radius == 1:
        # Tetracode (9 words of length 4) x all 3^2 tails -> 81 words.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        codewords = [h + list(tail)
                     for h in tetracode
                     for tail in itertools.product(range(3), repeat=n - 4)]

        powers = [q ** (n - 1 - i) for i in range(n)]
        covered = _build_coverage(codewords, q, n, radius)

        # First pass: remove redundant codewords.
        codewords, covered = _remove_redundant(codewords, covered, q, n, radius, powers, deadline)

        # Ensure coverage (should already be full, but guard).
        if not np.all(covered):
            codewords, covered = _greedy_repair(codewords, covered, q, n, radius, powers, deadline)

        # Second pass: try to remove more redundancy after any repairs.
        if _now() < deadline:
            codewords, covered = _remove_redundant(codewords, covered, q, n, radius, powers, deadline)
            if not np.all(covered):
                codewords, covered = _greedy_repair(codewords, covered, q, n, radius, powers, deadline)

        # Final safety: guarantee full coverage before returning.
        if not np.all(covered):
            codewords, covered = _greedy_repair(codewords, covered, q, n, radius, powers, deadline)

        return codewords

    # Generic fallback: brute force is infeasible for large spaces; use a simple
    # grid-based construction that is valid (covers everything) but not minimal.
    # Cover by choosing one word per "line" along each coordinate: place a codeword
    # at every word where all coordinates are 0 except we step through a lattice.
    # Simplest valid fallback: the full space (always valid, just large).
    # But to keep it small, use a product construction of radius-1 codes per coordinate
    # is not directly available; fall back to full space to guarantee validity.
    total = q ** n
    if total <= 200000:
        return [_idx_to_word(i, q, n) for i in range(total)]
    # For huge spaces, return a coarse grid that is guaranteed to cover:
    # choose words on a lattice with step such that every word is within radius.
    # A safe construction: all words where each coordinate is 0 (just the single
    # all-zero word) does NOT cover. Instead, use a covering by Hamming balls:
    # pick words spaced so balls of radius `radius` overlap. A simple valid set:
    # for each coordinate, include both 0 and (q-1) is not enough for radius 1 when q>2.
    # Safest: full space if it fits in memory/time, else a dense lattice.
    # Given the task is specifically q=3,n=6, this branch is rarely hit.
    # Use a lattice with step 1 along a diagonal to be safe: include all words with
    # sum of coordinates == 0 mod 1 -> all words. So just full space.
    return [_idx_to_word(i, q, n) for i in range(total)]
# EVOLVE-BLOCK-END
