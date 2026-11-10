"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def _is_dissociated_reciprocal(subset):
    """Check if all reciprocal subset sums are distinct using exact Fraction arithmetic.
    
    A set is dissociated in reciprocals iff no non-trivial subset sum of the
    reciprocals equals another non-trivial subset sum, which is equivalent to
    no subset of reciprocals summing to 0 (other than the empty set) when we
    consider differences.
    
    More precisely: all 2^n subset sums are distinct iff for any two distinct
    subsets A, B, sum(A) != sum(B), i.e., sum(A \\ B) != sum(B \\ A) where
    A \\ B and B \\ A are disjoint. This is equivalent to: no nonempty subset
    of the reciprocals, with signs ±1, sums to 0.
    
    We check by generating all 2^n subset sums and verifying uniqueness.
    For n up to ~14, this is 16384 sums which is fast.
    """
    n = len(subset)
    if n == 0:
        return True
    recips = [Fraction(1, x) for x in subset]
    
    # Generate all 2^n subset sums
    sums = [Fraction(0)]
    for r in recips:
        sums = sums + [s + r for s in sums]
    
    # Check all sums are distinct
    if len(set(sums)) != len(sums):
        return False
    return True


def _greedy_construct(order, deadline):
    """Greedy: add elements in the given order if dissociated property holds."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        candidate = selected + [x]
        if _is_dissociated_reciprocal(candidate):
            selected = candidate
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Seed solution (9 elements)
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    universe = list(range(1, 23))
    
    # Try multiple orderings for greedy construction
    orderings = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        # Try starting with larger elements first (they have smaller reciprocals, less collision)
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Mixed: alternate large and small
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 1],
        # Large primes first
        [19, 17, 13, 11, 7, 5, 3, 2, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4, 1],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_construct(order, deadline)
        if len(result) > len(best):
            best = result
            if len(best) >= 14:
                break  # Diminishing returns; 2^14 = 16384 checks per element
    
    # Local search: try to improve by swapping or adding
    if time.time() < deadline:
        # Try to add any missing element
        current_set = set(best)
        for x in universe:
            if time.time() > deadline:
                break
            if x not in current_set:
                candidate = best + [x]
                if _is_dissociated_reciprocal(candidate):
                    best = candidate
                    current_set.add(x)
    
    # Try swaps: replace one element with another to potentially allow more additions
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for x in universe:
                    if x in best:
                        continue
                    candidate = best[:i] + best[i+1:] + [x]
                    if _is_dissociated_reciprocal(candidate):
                        # Check if this allows adding more
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
    
    # Final: try to add elements again after swaps
    if time.time() < deadline:
        current_set = set(best)
        for x in universe:
            if time.time() > deadline:
                break
            if x not in current_set:
                candidate = best + [x]
                if _is_dissociated_reciprocal(candidate):
                    best = candidate
                    current_set.add(x)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
