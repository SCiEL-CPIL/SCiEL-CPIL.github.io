import time
from math import gcd
from fractions import Fraction

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums."""
    if not S:
        return True
    n = len(S)
    sums = [(0, 1)]
    for a in S:
        new_sums = []
        for (num, den) in sums:
            new_num = num * a + den
            new_den = den * a
            g = gcd(new_num, new_den)
            new_sums.append((new_num // g, new_den // g))
        sums.extend(new_sums)
    if len(set(sums)) != len(sums):
        return False
    return True

def _is_dissociated_fast(S):
    """
    Fast dissociation check using integer arithmetic.
    For a set S of integers, we check if all subset sums of {1/x : x in S} are distinct.
    This is equivalent to checking if all subset sums of {L/x : x in S} are distinct,
    where L is the LCM of all elements in S (or a common multiple).
    
    Actually, a simpler approach: use the fact that two subset sums are equal iff
    sum_{i in A} 1/i = sum_{j in B} 1/j for distinct subsets A, B.
    Multiply both sides by the LCM of all elements involved.
    
    For efficiency, we can precompute the LCM of all elements 1..22 and use that.
    """
    if not S:
        return True
    n = len(S)
    if n > 20:
        return False
    
    # Use a large common denominator to convert to integers
    # LCM of 1..22 is 232792560
    LCM_VAL = 232792560
    
    # Convert each element x to LCM_VAL // x
    coeffs = [LCM_VAL // x for x in S]
    
    # Generate all subset sums
    sums = [0]
    for c in coeffs:
        new_sums = [s + c for s in sums]
        sums.extend(new_sums)
    
    # Check for duplicates
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    deadline = time.time() + 80
    
    # LCM of 1..22
    LCM_VAL = 232792560
    
    # Integer coefficients: c[i] = LCM_VAL // (i+1) for element i+1
    c = [LCM_VAL // (i+1) for i in range(22)]
    
    # Precompute which pairs of subsets can have equal sums
    # For a fixed set S, two subsets A and B have equal sums iff
    # sum_{i in A} c[i] = sum_{j in B} c[j]
    
    best = []
    
    # Quick greedy to get a starting point
    def greedy_add(order):
        current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            if _is_dissociated_fast(test):
                current = test
        return current
    
    # Try several greedy orders
    import random
    random.seed(42)
    
    # Ascending order (small numbers first)
    result = greedy_add(list(range(1, 23)))
    if len(result) > len(best):
        best = result
    
    # Descending order (large numbers first)
    result = greedy_add(list(range(22, 0, -1)))
    if len(result) > len(best):
        best = result
    
    # Try some random orders
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_add(order)
        if len(result) > len(best):
            best = result
    
    # Now try to improve via local search / branch and bound
    # Strategy: try to add one more element by removing some others
    
    def can_add_to(S, a):
        """Check if we can add element a to set S."""
        if a in S:
            return False
        return _is_dissociated_fast(S + [a])
    
    # Local search: try to find a larger set
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
        
        current_size = len(best)
        best_set = set(best)
        
        # Try adding each unused element
        for a in range(1, 23):
            if time.time() > deadline - 5:
                break
            if a in best_set:
                continue
            if can_add_to(best, a):
                best = best + [a]
                best_set.add(a)
                improved = True
        
        # Try swapping: remove one element, add another
        if not improved:
            for i, a in enumerate(best):
                if time.time() > deadline - 5:
                    break
                # Try removing a and adding some other element
                reduced = best[:i] + best[i+1:]
                for b in range(1, 23):
                    if b in set(reduced):
                        continue
                    test = reduced + [b]
                    if _is_dissociated_fast(test):
                        best = test
                        improved = True
                        break
    
    # Try branch and bound for larger sets
    def bb_search(available, current, target_size):
        nonlocal best
        if time.time() > deadline - 2:
            return False
        if len(current) + len(available) < target_size:
            return False
        if len(current) == target_size:
            if _is_dissociated_fast(current):
                if len(current) > len(best):
                    best = current[:]
                return True
            return False
        if not available:
            return False
        
        # Try including available[0]
        a = available[0]
        rest = available[1:]
        
        # Check if current + [a] is still potentially valid
        if _is_dissociated_fast(current + [a]):
            if bb_search(rest, current + [a], target_size):
                return True
        
        # Try excluding available[0]
        if len(current) + len(rest) >= target_size:
            if bb_search(rest, current, target_size):
                return True
        
        return False
    
    # Try to find sets of size len(best) + 1
    for target in range(len(best) + 1, min(len(best) + 3, 23)):
        if time.time() > deadline - 10:
            break
        
        # Order elements: try larger numbers first (they're less likely to cause conflicts)
        bb_order = list(range(22, 0, -1))
        if bb_search(bb_order, [], target):
            pass  # Found a valid set of this size
        else:
            # Try ascending order too
            if time.time() > deadline - 10:
                break
            bb_order = list(range(1, 23))
            if bb_search(bb_order, [], target):
                pass
            else:
                break
    
    # Final verification
    if best and not _is_dissociated(best):
        # Fallback: use greedy result
        best = greedy_add(list(range(1, 23)))
    
    return best
