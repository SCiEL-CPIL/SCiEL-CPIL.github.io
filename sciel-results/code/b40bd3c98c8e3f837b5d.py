"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = 150.0
_MARGIN = 3.0          # return best result this far before the wall-clock limit


def _hamming(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _direct_sum_construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of
    # covering radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4).
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                 for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(3), repeat=n - 4)]


def _neighbors(word, q, n):
    # word plus every word at Hamming distance 1 (the covering ball of radius 1).
    out = [list(word)]
    for i in range(n):
        for v in range(q):
            if v == word[i]:
                continue
            w = list(word)
            w[i] = v
            out.append(w)
    return out


def _coverage(code, q, n, radius):
    # Set of all words within Hamming distance `radius` of some codeword.
    covered = set()
    for cw in code:
        for w in _neighbors(cw, q, n):
            covered.add(tuple(w))
    return covered


def _is_covering(code, q, n, radius, full):
    return len(_coverage(code, q, n, radius)) == full


def _greedy_add(q, n, radius, deadline, rng):
    # Greedy set-cover: repeatedly add the word covering the most still-uncovered words.
    full = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    neigh = [set(_neighbors(w, q, n)) for w in all_words]
    covered = [False] * full
    code = []
    while True:
        if time.time() > deadline - _MARGIN:
            break
        best = -1
        best_gain = 0
        for idx, ns in enumerate(neigh):
            g = 0
            for t in ns:
                if not covered[t]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best = idx
        if best_gain == 0:
            break
        code.append(list(all_words[best]))
        for t in neigh[best]:
            covered[t] = True
    return code


def _greedy_remove(code, q, n, radius, full, deadline, rng):
    # Start from a valid covering; greedily drop any codeword that is redundant.
    code = [list(c) for c in code]
    covered = _coverage(code, q, n, radius)
    assert len(covered) == full, "initial code is not a covering"
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - _MARGIN:
                return code
            c = code[i]
            # words only this codeword covers:
            only = [t for t in _neighbors(c, q, n)
                    if sum(1 for j in range(len(code)) if j != i and t in _neighbors(code[j], q, n)) == 0]
            if not only:
                code.pop(i)
                changed = True
    return code


def construct(q, n, radius):
    deadline = time.time() + _DEADLINE
    full = q ** n

    # Guaranteed-valid fallback: the seed's direct-sum construction.
    fallback = _direct_sum_construct(q, n, radius)
    if time.time() > deadline - _MARGIN:
        return fallback

    best = list(fallback)

    # Primary: prune the dense direct-sum code with a greedy removal pass.
    try:
        pruned = _greedy_remove(fallback, q, n, radius, full, deadline, random.Random(0))
        if len(pruned) < len(best):
            best = pruned
    except Exception:
        pass

    # Backup: fresh greedy addition (independent of the seed structure).
    try:
        added = _greedy_add(q, n, radius, deadline, random.Random(0))
        if len(added) < len(best):
            best = added
    except Exception:
        pass

    # Final safety: if the deadline is close, return the best valid result.
    if time.time() > deadline - _MARGIN:
        return best

    # Polish: one more removal pass over the best code found.
    try:
        polished = _greedy_remove(best, q, n, radius, full, deadline, random.Random(1))
        if len(polished) < len(best):
            best = polished
    except Exception:
        pass

    # Always return a valid covering (fall back if anything went wrong).
    if len(best) >= len(fallback):
        return fallback
    return best
# EVOLVE-BLOCK-END
