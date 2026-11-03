"""Improved covering-code search: vectorized greedy + local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin
_DEADLINE_SECONDS = 140.0


def _elapsed(start):
    return time.monotonic() - start


def _idx_to_tuple(idx, n, q):
    """Convert flat index to tuple of digits."""
    t = []
    for i in range(n):
        t.append(idx % q)
        idx //= q
    return t[::-1]


def _tuple_to_idx(t, n, q):
    """Convert tuple of digits to flat index."""
    idx = 0
    for d in t:
        idx = idx * q + d
    return idx


def _all_neighbors(codeword, n, q, radius):
    """Yield all words within Hamming distance <= radius of codeword."""
    # For radius 1: the word itself plus all single-position changes
    yield tuple(codeword)
    for pos in range(n):
        orig = codeword[pos]
        for val in range(q):
            if val == orig:
                continue
            new = list(codeword)
            new[pos] = val
            yield tuple(new)


def _greedy_construct(q, n, radius, rng, max_time_remaining):
    """
    Vectorized greedy set cover.
    Returns list of codewords (tuples) that cover all of {0..q-1}^n.
    """
    total = q ** n
    # Precompute: for each position and value, the set of words affected
    # Actually, let's use a different approach: maintain coverage as a boolean array
    covered = np.zeros(total, dtype=np.uint8)
    codewords = []

    # Precompute neighbor masks for each possible codeword? That's too large.
    # Instead, for each candidate codeword, compute which words it covers.
    # We'll iterate over all possible codewords and track uncovered count.

    # For efficiency, precompute for each position the "step" in the flat index
    # Changing digit at position pos from v to v' changes index by (v'-v) * q^(n-1-pos)
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    # For radius=1, a codeword covers: itself + for each pos, for each other val, the word with that val
    # Number of words covered by one codeword: 1 + n*(q-1)

    # Strategy: repeatedly pick the codeword that covers the most uncovered words
    # To do this efficiently, maintain for each codeword its current "gain"
    # But storing gain for all q^n codewords is 2187 ints - fine!

    # gain[c] = number of currently uncovered words within distance <= radius of codeword c
    gain = np.zeros(total, dtype=np.int32)

    # Initialize gain: for each codeword, count how many of its neighbors are uncovered (all are initially)
    # For radius 1: each codeword covers 1 + n*(q-1) words
    initial_gain = 1 + n * (q - 1)
    gain[:] = initial_gain

    # To update gains when we add a codeword, we need to know for each other codeword
    # how many of its neighbors overlap with the newly covered set.
    # This is expensive. Alternative: just recompute gain from scratch each iteration? No.
    # Better: for each newly covered word w, find all codewords c that have w as a neighbor,
    # and decrement gain[c] by 1.

    # Precompute: for each word w, the list of codewords c such that dist(c,w) <= radius
    # For radius 1: w is a neighbor of c iff c == w or dist(c,w)=1
    # Number of such c per w: 1 + n*(q-1) = same as above

    # Build reverse map: for each word w, list of codewords covering it
    # This is a list of lists, total size = total * (1 + n*(q-1))
    # For q=3, n=7: 2187 * (1+14) = 2187*15 = 32805 entries - fine

    reverse_cover = [[] for _ in range(total)]
    for c in range(total):
        c_tuple = _idx_to_tuple(c, n, q)
        # Add c itself
        reverse_cover[c].append(c)
        # Add single-position changes
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                w = _tuple_to_idx(new_tuple, n, q)
                reverse_cover[w].append(c)

    # Now greedy loop
    while True:
        elapsed_check = _elapsed(max_time_remaining[0])
        if elapsed_check > max_time_remaining[1]:
            break

        # Find codeword with max gain
        if np.max(gain) <= 0:
            break
        c = int(np.argmax(gain))
        codewords.append(_idx_to_tuple(c, n, q))

        # Mark all neighbors of c as covered
        c_tuple = _idx_to_tuple(c, n, q)
        to_cover = [c]
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                to_cover.append(_tuple_to_idx(new_tuple, n, q))

        for w in to_cover:
            if covered[w] == 0:
                covered[w] = 1
                # For each codeword that covers w, decrement its gain
                for cc in reverse_cover[w]:
                    gain[cc] -= 1

        # Check if all covered
        if covered.all():
            break

    return codewords


def _is_cover(codewords, q, n, radius):
    """Check if codewords form a valid covering."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        covered[c_idx] = 1
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_cw = list(cw)
                new_cw[pos] = val
                w_idx = _tuple_to_idx(new_cw, n, q)
                covered[w_idx] = 1

    return covered.all()


def _local_search(codewords, q, n, radius, rng, max_time_remaining):
    """
    Try to remove codewords one at a time if coverage is maintained.
    Then try swaps: remove one, add another if it reduces count or maintains.
    """
    codewords = list(codewords)
    total = q ** n
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    def _coverage_count(cw_list):
        """Count how many words are covered by cw_list."""
        covered = np.zeros(total, dtype=np.uint8)
        for cw in cw_list:
            c_idx = _tuple_to_idx(cw, n, q)
            covered[c_idx] = 1
            for pos in range(n):
                orig = cw[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_cw = list(cw)
                    new_cw[pos] = val
                    w_idx = _tuple_to_idx(new_cw, n, q)
                    covered[w_idx] = 1
        return int(np.sum(covered))

    # Phase 1: try to remove each codeword
    improved = True
    while improved and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        improved = False
        for i in range(len(codewords)):
            if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
                return codewords
            test_list = codewords[:i] + codewords[i+1:]
            if _coverage_count(test_list) == total:
                codewords = test_list
                improved = True
                break

    # Phase 2: try swaps - remove one, add a different one if it still covers all
    # This is expensive; do a limited number of random attempts
    swaps_tried = 0
    max_swaps = 500
    while swaps_tried < max_swaps and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        i = rng.integers(0, len(codewords))
        # Pick a random candidate to add
        cand_idx = rng.integers(0, total)
        cand = _idx_to_tuple(cand_idx, n, q)
        if cand in codewords:
            swaps_tried += 1
            continue
        test_list = codewords[:i] + codewords[i+1:] + [cand]
        if _coverage_count(test_list) == total:
            codewords = test_list
            # Reset swap counter since we made progress
            swaps_tried = 0
        else:
            swaps_tried += 1

    return codewords


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    max_time_remaining = [start, deadline]

    rng = np.random.default_rng(42)

    # Start with a greedy construction
    best = _greedy_construct(q, n, radius, rng, max_time_remaining)

    # Local search to improve
    if _elapsed(start) < deadline - 5.0:
        best = _local_search(best, q, n, radius, rng, max_time_remaining)

    # Verify coverage
    if not _is_cover(best, q, n, radius):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
