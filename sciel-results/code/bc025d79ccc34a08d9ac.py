"""Rewritten program: exhaustive/structured search for A,B with max distinct pairwise-coprime sums."""
import time
import math
import itertools
import sys

_DEADLINE = 60.0  # safety margin before 90s kill

def _gcd(a, b):
    return math.gcd(a, b)

def _pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score_pair(a, b):
    # a, b are lists of 3 distinct ints in [1,128]
    sums = [x + y for x in a for y in b]
    distinct = list(set(sums))
    if not _pairwise_coprime(distinct):
        return -1, 0
    return len(distinct), sums

def construct():
    t0 = time.time()
    best = 7  # seed baseline
    best_ab = [[1, 7, 13], [4, 10, 16]]

    # Precompute all 3-combinations of [1,128]
    rng = range(1, 129)
    # To keep search tractable, we will iterate over B combinations and A combinations
    # but prune early. C(128,3) = 341,920. Full double loop is ~1.17e11 — too slow.
    # Instead: iterate over B (341,920) and for each, iterate over A but with early
    # exit and a smarter inner loop. Still too slow for full.
    #
    # Better approach: Use the fact that sums must be pairwise coprime.
    # For a fixed B = {b1,b2,b3}, the 9 sums are a_i + b_j.
    # We can iterate A and check. But 341920 * 341920 is too many.
    #
    # Alternative: Randomized + local search with a good heuristic.
    # Or: iterate over B, and for each B, do a smarter search over A.
    #
    # Let's try: for each B, iterate over A combinations but skip if any two sums
    # share a factor. Early pruning.
    #
    # Actually, let's do a randomized search first, then local improvement.

    import random
    random.seed(42)

    def random_trial():
        A = random.sample(rng, 3)
        B = random.sample(rng, 3)
        s, _ = _score_pair(A, B)
        return s, A, B

    # Phase 1: Random sampling
    trials = 0
    while time.time() - t0 < _DEADLINE * 0.5 and trials < 2000000:
        s, A, B = random_trial()
        if s > best:
            best = s
            best_ab = [A, B]
        trials += 1
        if best >= 9:
            break

    if best >= 9:
        return best_ab

    # Phase 2: Structured search over B, with limited A search per B
    # For each B, try many random A's
    elapsed = time.time() - t0
    remaining = _DEADLINE - elapsed
    if remaining < 1:
        return best_ab

    all_B = list(itertools.combinations(rng, 3))
    # Shuffle for variety
    random.shuffle(all_B)

    b_idx = 0
    a_trials_per_b = 200
    while b_idx < len(all_B) and time.time() - t0 < _DEADLINE:
        B = list(all_B[b_idx])
        for _ in range(a_trials_per_b):
            if time.time() - t0 > _DEADLINE:
                break
            A = random.sample(rng, 3)
            s, _ = _score_pair(A, B)
            if s > best:
                best = s
                best_ab = [A, B]
                if best >= 9:
                    return best_ab
        b_idx += 1

    return best_ab
