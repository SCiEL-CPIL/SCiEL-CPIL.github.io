"""Improved covering-code search for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 150.0  # wall-clock seconds
_MARGIN = 2.0      # safety margin


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _within_radius(word, code, radius):
    return any(_hamming(word, c) <= radius for c in code)


def _is_covering(code, q, n, radius):
    # Full check over all q^n words
    for word in itertools.product(range(q), repeat=n):
        if not _within_radius(word, code, radius):
            return False
    return True


def _base_tetracode_sum(q, n, radius):
    # Direct sum of ternary tetracode with full space on remaining coords
    if q != 3 or n < 4:
        # Fallback: greedy
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = [h + list(tail) for h in tetracode
             for tail in itertools.product(range(3), repeat=n - 4)]
    return words


def _greedy_cover(q, n, radius, start_code=None, time_limit=30.0):
    """Greedy set-cover: repeatedly add the word covering the most uncovered words."""
    t0 = time.time()
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    # Precompute for each candidate codeword, the set of words it covers
    # For efficiency, represent coverage as a numpy bool array
    if start_code is None:
        start_code = []

    covered = np.zeros(N, dtype=bool)
    # Mark words covered by initial code
    for c in start_code:
        for i, w in enumerate(all_words):
            if sum(x != y for x, y in zip(c, w)) <= radius:
                covered[i] = True

    code = list(start_code)

    while not covered.all() and (time.time() - t0) < time_limit:
        # Find the best candidate: try all words as potential new codewords
        # For speed, only consider uncovered words and their neighbors as candidates
        # Actually, a codeword can be any word; best is to pick the one maximizing new coverage
        # We'll sample candidates to keep it fast
        best_gain = 0
        best_word = None

        # Candidate pool: all words (but we can restrict to uncovered + their neighbors)
        # For n=6, q=3, N=729, checking all is feasible
        for c in all_words:
            if covered.all():
                break
            gain = 0
            for i in range(N):
                if not covered[i]:
                    if sum(x != y for x, y in zip(c, all_words[i])) <= radius:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = c
            if (time.time() - t0) > time_limit:
                break

        if best_gain == 0:
            break

        code.append(list(best_word))
        # Update covered
        for i in range(N):
            if not covered[i]:
                if sum(x != y for x, y in zip(best_word, all_words[i])) <= radius:
                    covered[i] = True

    return code


def _local_search_improve(code, q, n, radius, time_limit=20.0):
    """Try to remove or swap codewords to reduce size while maintaining coverage."""
    t0 = time.time()
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    code = [list(c) for c in code]
    improved = True
    while improved and (time.time() - t0) < time_limit:
        improved = False
        # Try removing each codeword
        for i in range(len(code)):
            if (time.time() - t0) > time_limit:
                break
            remaining = [c for j, c in enumerate(code) if j != i]
            if _is_covering(remaining, q, n, radius):
                code = remaining
                improved = True
                break
        # Try swapping: remove one, add another
        if not improved and (time.time() - t0) < time_limit:
            for i in range(len(code)):
                if (time.time() - t0) > time_limit:
                    break
                remaining = [c for j, c in enumerate(code) if j != i]
                # Find if any word covers all words that only code[i] covered
                # For now, just try adding random words
                for _ in range(50):
                    new_word = list(random.choice(all_words))
                    candidate = remaining + [new_word]
                    if len(candidate) < len(code):
                        continue
                    if _is_covering(candidate, q, n, radius):
                        code = candidate
                        improved = True
                        break
                if improved:
                    break
    return code


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # Start with the known construction
    code = _base_tetracode_sum(q, n, radius)
    if code is None:
        code = []

    # Verify it's covering
    if not _is_covering(code, q, n, radius):
        # Fallback to pure greedy
        code = _greedy_cover(q, n, radius, start_code=[], time_limit=60.0)

    # Try local search to improve (reduce) the code
    if time.time() < deadline:
        code = _local_search_improve(code, q, n, radius, time_limit=min(30.0, deadline - time.time()))

    # Try greedy from scratch with a time limit and compare
    if time.time() < deadline:
        greedy_code = _greedy_cover(q, n, radius, start_code=[], time_limit=min(40.0, deadline - time.time()))
        if greedy_code and len(greedy_code) < len(code):
            code = greedy_code
            # Try to improve the greedy result
            if time.time() < deadline:
                code = _local_search_improve(code, q, n, radius, time_limit=min(20.0, deadline - time.time()))

    # Final check
    if not _is_covering(code, q, n, radius):
        # Emergency: return the base construction if it's valid
        base = _base_tetracode_sum(q, n, radius)
        if base is not None and _is_covering(base, q, n, radius):
            return base
        # Otherwise, do a quick greedy
        return _greedy_cover(q, n, radius, start_code=[], time_limit=30.0)

    return code
# EVOLVE-BLOCK-END
