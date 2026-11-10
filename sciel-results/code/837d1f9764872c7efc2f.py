"""Large integer sets avoiding equal-gcd triples.

Strategy: start from the set of all odd numbers in [1,128] (64 elements),
which is large and structurally avoids most equal-gcd triples. Then attempt
to add even numbers one at a time (with a repair step) whenever doing so does
not create a bad triple. Finally, run a local-search pass to try to grow the
set further.
"""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin under the 90s wall-clock kill


def _bad_triple(a, b, c):
    """True if the three pairwise gcds of a,b,c are all equal."""
    return math.gcd(a, b) == math.gcd(a, c) == math.gcd(b, c)


def _is_valid(S):
    """True if no three distinct elements of S form an equal-gcd triple."""
    for a, b, c in combinations(S, 3):
        if _bad_triple(a, b, c):
            return False
    return True


def _bad_triples_in(S):
    """Return the list of bad triples within S."""
    bad = []
    for a, b, c in combinations(S, 3):
        if _bad_triple(a, b, c):
            bad.append((a, b, c))
    return bad


def _conflict_with_new(S, x):
    """Count how many pairs (a,b) in S would form a bad triple with x."""
    cnt = 0
    for a, b in combinations(S, 2):
        if _bad_triple(a, b, x):
            cnt += 1
    return cnt


def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal. Returns the best set found."""
    random.seed(20240611)
    full = list(range(1, 129))

    def within_deadline():
        return (time.time() - _START_TIME) < _DEADLINE - 1.5

    # ---- Base: all odd numbers in [1,128] ----
    base = [x for x in full if x % 2 == 1]
    # Verify base validity (should be valid)
    if not _is_valid(base):
        # Fallback: greedy removal
        S = list(base)
        bad = _bad_triples_in(S)
        while bad and within_deadline():
            # Remove the element that appears in the most bad triples
            from collections import Counter
            c = Counter()
            for t in bad:
                for v in t:
                    c[v] += 1
            v = c.most_common(1)[0][0]
            S = [x for x in S if x != v]
            bad = _bad_triples_in(S)
        base = S

    best_set = list(base)
    best_size = len(base)

    # ---- Greedy addition of even numbers ----
    # Try to add even numbers in an order that minimizes conflict.
    evens = [x for x in full if x % 2 == 0]

    def try_add_evens(order):
        S = list(best_set)
        for x in order:
            if not within_deadline():
                break
            if x in S:
                continue
            # Check if x can be added without creating any bad triple
            if _conflict_with_new(S, x) == 0:
                S.append(x)
        return S

    # Several orders to try
    orders = [
        evens,                                  # ascending
        list(reversed(evens)),                  # descending
        sorted(evens, key=lambda x: x.bit_count()),  # fewer bits first
        sorted(evens, key=lambda x: -x.bit_count()), # more bits first
        list(evens),
    ]
    random.shuffle(orders[4])  # one random order

    for order in orders:
        if not within_deadline():
            break
        cand = try_add_evens(order)
        if len(cand) > best_size:
            best_size = len(cand)
            best_set = cand

    # ---- Local search: swap an element out and a new one in to grow ----
    def local_search(S, iters=2000):
        S = list(S)
        cur_size = len(S)
        for _ in range(iters):
            if not within_deadline():
                break
            # Try to add a random element not in S
            candidates = [x for x in full if x not in S]
            if not candidates:
                break
            x = random.choice(candidates)
            conf = _conflict_with_new(S, x)
            if conf == 0:
                S.append(x)
                cur_size = len(S)
                continue
            # If conflict, try removing one of the conflicting elements
            # Find a conflicting pair
            # (skip heavy work; just try a random removal from the set and see if x fits)
            # Pick a random element to drop
            if S:
                v = random.choice(S)
                S2 = [y for y in S if y != v]
                if _conflict_with_new(S2, x) == 0:
                    S2.append(x)
                    if len(S2) > cur_size:
                        S = S2
                        cur_size = len(S)
        return S

    if within_deadline():
        for _ in range(10):
            if not within_deadline():
                break
            cand = local_search(best_set, iters=1500)
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = cand

    # ---- Final safety: verify and, if invalid, repair by removing offenders ----
    if not _is_valid(best_set):
        S = list(best_set)
        bad = _bad_triples_in(S)
        while bad and within_deadline():
            from collections import Counter
            c = Counter()
            for t in bad:
                for v in t:
                    c[v] += 1
            v = c.most_common(1)[0][0]
            S = [x for x in S if x != v]
            bad = _bad_triples_in(S)
        best_set = S

    if not best_set:
        return [1]

    return best_set

# EVOLVE-BLOCK-END
