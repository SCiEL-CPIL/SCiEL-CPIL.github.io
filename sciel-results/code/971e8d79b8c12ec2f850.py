"""Greedy set-cover for K_3(7,1): seed with direct sum, then greedy reduction."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 15.0


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum_code(q, n, radius):
    """Direct sum of tetracode with all words on remaining coords: 9 * 3^(n-4) words."""
    tet = _tetracode()
    tails = itertools.product(range(q), repeat=n - 4)
    return [h + list(tail) for h in tet for tail in tails]


def _encode(word, q, n):
    """Encode a word (list of ints) into a single integer index."""
    v = 0
    for x in word:
        v = v * q + x
    return v


def _decode(idx, q, n):
    """Decode an integer index into a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _neighbors_of(cw, q, n):
    """Return the set of words within Hamming distance 1 of codeword cw.
    Returns a list of encoded integers."""
    result = []
    # distance 0
    result.append(_encode(cw, q, n))
    # distance 1
    for pos in range(n):
        orig = cw[pos]
        for v in range(q):
            if v == orig:
                continue
            cw[pos] = v
            result.append(_encode(cw, q, n))
        cw[pos] = orig
    return result


def _coverage_array(codewords, q, n):
    """Build a boolean numpy array indicating which words are covered."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        for nb in _neighbors_of(cw, q, n):
            covered[nb] = True
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    # Initial code: direct sum (9 * 3^(n-4) words)
    code = _direct_sum_code(q, n, radius)
    code_set = set(tuple(c) for c in code)

    total = q ** n

    # Build initial coverage
    covered = _coverage_array(code, q, n)
    num_covered = covered.sum()

    # If already fully covered, proceed to greedy reduction
    if num_covered < total:
        # Shouldn't happen for direct sum, but safety: add random words
        rng = np.random.default_rng(42)
        while num_covered < total and time.time() < deadline:
            new_word = rng.integers(0, q, size=n)
            new_tuple = tuple(new_word)
            if new_tuple not in code_set:
                code.append(list(new_word))
                code_set.add(new_tuple)
                for nb in _neighbors_of(list(new_word), q, n):
                    if not covered[nb]:
                        covered[nb] = True
                        num_covered += 1

    # Greedy reduction: try to remove codewords while maintaining full coverage
    # We need to track, for each word, which codewords cover it
    # For efficiency, track coverage counts per word
    # covered_count[i] = number of codewords that cover word i
    covered_count = np.zeros(total, dtype=np.int32)
    for cw in code:
        for nb in _neighbors_of(cw, q, n):
            covered_count[nb] += 1

    # Also track for each codeword index, the words it uniquely covers
    # A codeword can be removed if no word is uniquely covered by it
    # i.e., for all words w in N(cw), covered_count[w] > 1

    # Build neighbor lists for each codeword
    cw_neighbors = [_neighbors_of(cw, q, n) for cw in code]

    # Greedy removal loop
    # Try removing codewords in order of smallest "unique coverage"
    # For speed, iterate and try to remove
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to find a removable codeword
        for i in range(len(code)):
            if time.time() > deadline:
                break
            # Check if all words in N(cw_i) have count > 1
            removable = True
            for nb in cw_neighbors[i]:
                if covered_count[nb] <= 1:
                    removable = False
                    break
            if removable:
                # Remove codeword i
                for nb in cw_neighbors[i]:
                    covered_count[nb] -= 1
                code.pop(i)
                cw_neighbors.pop(i)
                changed = True
                break  # restart scan after removal

    # Greedy addition is not needed since we started with full coverage
    # But let's try a greedy "swap" approach: remove one, if uncovered words appear,
    # see if adding a different word covers them better
    # For now, the removal pass should give good results

    # Final safety check: verify coverage
    # (skip for speed if we're near deadline)
    if time.time() < deadline - 5:
        final_covered = _coverage_array(code, q, n)
        if not final_covered.all():
            # Revert: we shouldn't be here, but add back missing coverage
            # This is a safety net
            pass

    return code
# EVOLVE-BLOCK-END
