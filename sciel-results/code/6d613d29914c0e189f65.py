"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from fractions import Fraction

# Precompute reciprocals as Fractions for exact arithmetic
UNIVERSE = list(range(1, 23))
RECIPS = {i: Fraction(1, i) for i in UNIVERSE}

def _all_subset_sums_distinct(subset):
    """Check if all subset sums of reciprocals are distinct (including empty sum=0)."""
    n = len(subset)
    sums = set()
    # Use iterative subset sum enumeration
    # For n up to ~12-14, 2^n is manageable
    sums.add(Fraction(0, 1))
    for i in range(n):
        r = RECIPS[subset[i]]
        # Add r to all existing sums
        current_sums = list(sums)
        for s in current_sums:
            ns = s + r
            if ns in sums:
                return False
            sums.add(ns)
    return True

def _score(subset):
    """Score is cardinality if valid, 0 otherwise."""
    if not subset:
        return 0
    if _all_subset_sums_distinct(subset):
        return len(subset)
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # seed: 9 elements
    best_score = 9
    
    # Strategy: Greedy construction + local search to maximize cardinality
    # First, try greedy approach with different orderings
    # Then, try to add elements to the seed
    
    # Try greedy: start with smallest, add largest possible
    def greedy_greedy():
        chosen = []
        remaining = list(UNIVERSE)
        for i in range(len(remaining)):
            # Try adding each remaining element, prefer larger ones for diversity
            # Actually, for dissociated sets, we want elements that don't create collisions
            # Greedy: try to add elements in decreasing order
            pass
    
    # Better: Start with seed and try to add elements
    # Also try different seeds via greedy
    
    # Greedy construction: try to build a large set
    # Key insight: For reciprocal sets, if we have elements a, b, c where 1/a + 1/b = 1/c, that's a collision
    # More generally, any linear relation with coefficients in {-1, 0, 1} with sum=0 is a collision
    
    # Let's use a greedy approach: start empty, add elements that don't create collisions
    # Try multiple random orderings and pick the best
    
    def greedy_build(order):
        chosen = []
        for x in order:
            candidate = chosen + [x]
            if _all_subset_sums_distinct(candidate):
                chosen = candidate
        return chosen
    
    # Try sorted descending (largest first)
    order_desc = sorted(UNIVERSE, reverse=True)
    s = greedy_build(order_desc)
    if len(s) > best_score:
        best_score = len(s)
        best_subset = s
    
    # Try sorted ascending
    order_asc = sorted(UNIVERSE)
    s = greedy_build(order_asc)
    if len(s) > best_score:
        best_score = len(s)
        best_subset = s
    
    # Try random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = UNIVERSE[:]
        random.shuffle(order)
        s = greedy_build(order)
        if len(s) > best_score:
            best_score = len(s)
            best_subset = s
    
    # Now try local search: add elements to best_subset
    def try_add_elements(subset):
        subset = list(subset)
        improved = True
        while improved:
            improved = False
            for x in UNIVERSE:
                if x not in subset:
                    candidate = subset + [x]
                    if _all_subset_sums_distinct(candidate):
                        subset = candidate
                        improved = True
                        break
        return subset
    
    # Try to extend best_subset
    extended = try_add_elements(best_subset)
    if len(extended) > best_score:
        best_score = len(extended)
        best_subset = extended
    
    # Try local search: swap elements to potentially get a larger set
    # For each element in best_subset, try removing it and adding two others
    if time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline:
                break
            subset = list(best_subset)
            random.shuffle(subset)
            # Try removing one element
            for i in range(len(subset)):
                if time.time() > deadline:
                    break
                test = subset[:i] + subset[i+1:]
                # Try adding pairs of elements not in test
                others = [x for x in UNIVERSE if x not in test]
                random.shuffle(others)
                for j in range(min(5, len(others))):
                    for k in range(j+1, min(6, len(others))):
                        candidate = test + [others[j], others[k]]
                        if len(candidate) > best_score:
                            if _all_subset_sums_distinct(candidate):
                                best_score = len(candidate)
                                best_subset = candidate
                                break
                    if len(best_subset) > best_score:
                        break
                if len(best_subset) > best_score:
                    break
            if len(best_subset) > best_score:
                break
    
    # Final check
    if not best_subset:
        best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_subset
# EVOLVE-BLOCK-END
