"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # 80s safety margin within 90s limit

    # We want a subset S of [1..24] such that for every x in S,
    # 1/x != sum_{y in T} 1/y for any subset T of S \ {x}.
    # Equivalently: no reciprocal in the set is a subset-sum of the reciprocals of the OTHER selected elements.

    # To maximize cardinality, we want to include as many elements as possible.
    # The constraint is: for each x in S, 1/x is NOT representable as a sum of distinct 1/y for y in S\{x}.

    # Key insight: larger elements have smaller reciprocals. If we pick large elements,
    # their reciprocals are small and hard to form sums of. But small elements have large reciprocals.
    # The dangerous case is when 1/x = sum of some 1/y's. This is easier when x is small (1/x large)
    # and the y's are also small.

    # Strategy: Use exact rational arithmetic (Fraction) to check validity.
    # Search for the maximum size valid subset.

    # Precompute reciprocals as Fractions
    N = 24
    recip = [Fraction(1, i) for i in range(1, N + 1)]
    # recip[i-1] = 1/i

    def is_valid(S):
        """Check if set S (list of 1-indexed values) is valid: no reciprocal equals subset-sum of others."""
        S_set = set(S)
        for x in S:
            target = Fraction(1, x)
            others = [y for y in S if y != x]
            # Check if target can be formed as sum of subset of reciprocals of others
            # Use subset-sum over the reciprocals
            if _subset_sum_exists(others, target, recip):
                return False
        return True

    def _subset_sum_exists(others, target, recip):
        """Check if target equals sum of reciprocals of some subset of others.
        others is list of values (1-indexed), recip[i-1]=1/i."""
        # Map value -> index for recip
        # Use DP with set of achievable sums (as Fractions)
        # Since |others| <= 24, but we can prune
        achievable = {Fraction(0, 1)}
        for y in others:
            r = recip[y - 1]
            new_sums = set()
            for s in achievable:
                ns = s + r
                if ns <= target:  # prune: all reciprocals positive
                    new_sums.add(ns)
            achievable |= new_sums
        return target in achievable

    # Search: try to find largest valid subset
    # Start with all 24, remove elements to fix violations
    # Or: greedy + local search

    # Let's do a systematic search: try all subsets of size k from 24 down,
    # but that's C(24,k) which is too large.

    # Better: greedy construction + local search
    # Greedy: add elements one by one (in some order), maintaining validity.

    # Try multiple orderings and pick best.

    best_set = None
    best_size = 0

    # Try greedy with different orderings
    # Order 1: descending (largest first) - likely safe since reciprocals are small
    # Order 2: ascending
    # Order 3: random permutations

    def greedy_construct(order):
        """Try to build valid set by adding elements in given order."""
        S = []
        for x in order:
            trial = S + [x]
            if is_valid(trial):
                S = trial
        return S

    # Try descending order
    desc_order = list(range(N, 0, -1))
    S1 = greedy_construct(desc_order)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Try ascending order
    asc_order = list(range(1, N + 1))
    S2 = greedy_construct(asc_order)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2

    # Try random permutations
    import random
    random.seed(42)
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        S = greedy_construct(perm)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Local search: try to add more elements to best_set
    if best_set is not None:
        S = list(best_set)
        S_set = set(S)
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for x in range(1, N + 1):
                if x not in S_set:
                    trial = S + [x]
                    if is_valid(trial):
                        S = trial
                        S_set.add(x)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S
                        break  # restart loop with new S

    # Also try: start from full set and remove minimal elements
    # This is a hitting set / deletion approach
    full = list(range(1, N + 1))
    if is_valid(full):
        return full

    # Try to find a large valid set via deletion from full
    # Identify violations and remove one element per violation
    def find_violation(S):
        """Return (x, T) if 1/x = sum(1/y for y in T) with T subset of S\{x}, else None."""
        S_set = set(S)
        for x in S:
            target = Fraction(1, x)
            others = [y for y in S if y != x]
            # Find subset T of others whose reciprocals sum to target
            T = _find_subset(others, target, recip)
            if T is not None:
                return (x, T)
        return None

    def _find_subset(others, target, recip):
        """Find subset of others whose reciprocals sum to target. Returns list of values or None."""
        # Backtracking search
        others_sorted = sorted(others, key=lambda y: -recip[y-1])  # larger reciprocals first
        def bt(idx, current_sum, current_set):
            if current_sum == target:
                return list(current_set)
            if idx >= len(others_sorted):
                return None
            r = recip[others_sorted[idx] - 1]
            if current_sum + r > target:
                return None
            # Include
            res = bt(idx + 1, current_sum + r, current_set + [others_sorted[idx]])
            if res is not None:
                return res
            # Exclude
            res = bt(idx + 1, current_sum, current_set)
            if res is not None:
                return res
            return None
        return bt(0, Fraction(0, 1), [])

    # Deletion approach: iteratively remove elements to fix violations
    S_del = list(range(1, N + 1))
    max_del_iters = 50
    for _ in range(max_del_iters):
        if time.time() > DEADLINE:
            break
        viol = find_violation(S_del)
        if viol is None:
            break
        x, T = viol
        # Remove one element from {x} ∪ T to break the violation
        candidates = [x] + list(T)
        removed = False
        for c in candidates:
            trial = [y for y in S_del if y != c]
            if is_valid(trial):
                S_del = trial
                removed = True
                break
        if not removed:
            # Remove x as fallback
            S_del = [y for y in S_del if y != x]

    if len(S_del) > best_size:
        best_size = len(S_del)
        best_set = S_del

    # Ensure we have a nonempty valid set
    if best_set is None or len(best_set) == 0:
        # Fallback: just {24}
        return [24]

    # Final validation
    if not is_valid(best_set):
        # Should not happen, but fallback
        # Try single elements
        for i in range(1, N + 1):
            if is_valid([i]):
                return [i]
        return [24]

    return best_set
# EVOLVE-BLOCK-END
