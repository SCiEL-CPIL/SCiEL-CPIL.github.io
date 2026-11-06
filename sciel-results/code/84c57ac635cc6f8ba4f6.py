<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
=======
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
    n = len(set_s)
    if n == 0:
        return True
    sums = {Fraction(0)}
    for x in set_s:
        new_sums = set()
        for s in sums:
            new_sums.add(s + Fraction(1, x))
        if len(new_sums) != len(sums):
            return False
        sums = new_sums
    return True


def _try_greedy(order):
    """Greedily add elements in given order, keeping only if no collision."""
    sums = {Fraction(0)}
    result = []
    for x in order:
        new_sums = set()
        for s in sums:
            new_sums.add(s + Fraction(1, x))
        if len(new_sums) == len(sums):
            sums = new_sums
            result.append(x)
    return result


def construct():
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # seed fallback
    universe = list(range(1, 23))
    
    # Try several random orderings
    random.seed(42)
    for trial in range(2000):
        if _deadline_hit():
            break
        order = universe[:]
        random.shuffle(order)
        candidate = _try_greedy(order)
        if len(candidate) > len(best):
            best = candidate
    
    # Also try some structured orderings
    for start_val in range(1, 23):
        if _deadline_hit():
            break
        # Try starting from different points and going up/down
        order = list(range(start_val, 23)) + list(range(1, start_val))
        candidate = _try_greedy(order)
        if len(candidate) > len(best):
            best = candidate
        
        order = list(range(start_val, 0, -1)) + list(range(start_val + 1, 23))
        candidate = _try_greedy(order)
        if len(candidate) > len(best):
            best = candidate
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
