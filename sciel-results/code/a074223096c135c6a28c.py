"""Rewritten program: exact cardinality-indexed subset-sum DP to maximize set size."""
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _max_cardinality_with_dp(elements):
    """
    Given a list of distinct positive integers, compute the maximum cardinality
    such that no two subsets of different cardinalities have the same sum.
    Uses DP tracking achievable sums per cardinality.
    Returns the maximum valid cardinality (0 means only empty set is safe, which is always true).
    Actually returns the max k such that all subset sums of size k are distinct from
    all subset sums of size != k. We need: for all i != j, sumset_i ∩ sumset_j = empty.
    
    We track for each cardinality c, the set of achievable sums.
    Then check pairwise disjointness.
    """
    n = len(elements)
    # sumsets[c] = set of sums achievable with exactly c elements
    # Start: sumsets[0] = {0}
    sumsets = [set() for _ in range(n + 1)]
    sumsets[0] = {0}
    
    for x in elements:
        # Update in reverse to avoid using x multiple times
        for c in range(n, 0, -1):
            if sumsets[c - 1]:
                for s in sumsets[c - 1]:
                    sumsets[c].add(s + x)
    
    # Now check: for all i != j, sumsets[i] ∩ sumsets[j] must be empty
    # If any collision exists at cardinality k, then we cannot have k in our set
    # Actually, we want the maximum number of elements we can keep.
    # But the problem says: "Whenever two subsets have different cardinalities their sums must differ."
    # This must hold for ALL subsets of the chosen set.
    # So if we have a set S, we need: for all i != j, sumset_S(i) ∩ sumset_S(j) = ∅.
    
    # The DP above computes sumsets for the full list.
    # But we need to find the largest subset of candidates that satisfies the property.
    # This is hard. Instead, let's use a greedy/backtracking approach.
    
    # For now, return whether the full set is valid
    for i in range(len(sumsets)):
        for j in range(i + 1, len(sumsets)):
            if sumsets[i] and sumsets[j] and sumsets[i] & sumsets[j]:
                return False
    return True


def _is_valid_set(elements):
    """Check if a set of elements satisfies the cardinality-separated subset sum property."""
    if not elements:
        return True
    n = len(elements)
    max_sum = sum(elements)
    # sumsets[c] = set of sums with exactly c elements
    sumsets = [set() for _ in range(n + 1)]
    sumsets[0] = {0}
    
    for x in elements:
        for c in range(n, 0, -1):
            if sumsets[c - 1]:
                for s in sumsets[c - 1]:
                    sumsets[c].add(s + x)
    
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sumsets[i] and sumsets[j]:
                if sumsets[i] & sumsets[j]:
                    return False
    return True


def construct():
    """
    Construct a set of distinct integers in 1..256 maximizing size such that
    no two subsets of different cardinalities have the same sum.
    """
    # We want to maximize the number of elements.
    # Let's try to find a large set.
    # Strategy: start with all numbers 1..256 and try to greedily add elements,
    # checking validity. Or use backtracking.
    
    # Given the 60s limit, let's try a greedy approach with some heuristics.
    # Start with empty set, try to add elements in some order.
    # Order: maybe larger elements first? Or smaller?
    # Let's try: start with elements in descending order, add if valid.
    
    # Actually, let's think about what sets work.
    # If we use powers of 2: 1, 2, 4, 8, ..., then all subset sums are unique,
    # so certainly sums of different cardinalities are different.
    # Powers of 2 up to 256: 1, 2, 4, 8, 16, 32, 64, 128 -> 8 elements.
    # Can we do better?
    
    # Let's try to construct a set greedily.
    # Start with an empty set, and try adding elements from 1 to 256.
    
    current = []
    for x in range(1, 257):
        candidate = current + [x]
        if _is_valid_set(candidate):
            current = candidate
    
    # Convert to list of Python ints
    result = [int(x) for x in current]
    return result

# EVOLVE-BLOCK-END
