"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated iff all 2^|S| subset sums of {1/x : x in S} are distinct.
We use greedy construction with exact Fraction arithmetic, trying multiple
random orderings and keeping the best valid set found before the deadline.
"""
import time
import random
from fractions import Fraction

_DEADLINE = 75.0  # safety margin before 90s evaluator kill
_START = time.time()


def _deadline_hit():
    return (time.time() - _START) > _DEADLINE


def _is_dissociated(set_s):
    """Check if all subset sums of reciprocals of set_s are distinct."""
    if not set_s:
        return True
    sums = {Fraction(0)}
    for x in set_s:
        new_sums = {s + Fraction(1, x) for s in sums}
        if not sums.isdisjoint(new_sums):
            return False
        sums.update(new_sums)
    return True


def _try_greedy(order):
    """Greedily add elements in given order, keeping only if no collision."""
    sums = {Fraction(0)}
    result = []
    for x in order:
        new_sums = {s + Fraction(1, x) for s in sums}
        if sums.isdisjoint(new_sums):
            sums.update(new_sums)
            result.append(x)
    return result


def construct():
    # Start with a small, verified valid set
    best = [1, 2, 3]
    if not _is_dissociated(best):
        best = [1] # Fallback to trivially valid
    
    universe = list(range(1, 23))
    
    # Try several random orderings
    random.seed(42)
    trials = 0
    while trials < 5000:
        if _deadline_hit():
            break
        trials += 1
        order = universe[:]
        random.shuffle(order)
        candidate = _try_greedy(order)
        if len(candidate) > len(best):
            best = candidate
            # Optional: if we find a very large set, we can stop early or continue
            if len(best) >= 15:
                break

    # Try structured orderings (descending often works well for reciprocals)
    if not _deadline_hit():
        order_desc = list(range(22, 0, -1))
        candidate = _try_greedy(order_desc)
        if len(candidate) > len(best):
            best = candidate
            
        order_asc = list(range(1, 23))
        candidate = _try_greedy(order_asc)
        if len(candidate) > len(best):
            best = candidate

    # Final verification
    if not _is_dissociated(best):
        # If somehow invalid, fall back to a known small valid set
        # {1, 2, 3, 5, 7, 11, 13} is likely valid but let's be safe
        best = [1, 2, 3]

    return best
# EVOLVE-BLOCK-END
