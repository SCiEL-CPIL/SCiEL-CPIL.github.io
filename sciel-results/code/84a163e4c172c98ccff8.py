import time
from itertools import combinations

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums."""
    if not S:
        return True
    n = len(S)
    # Compute all 2^n subset sums using fractions to avoid floating point issues
    # Use tuple of (numerator, denominator) in reduced form
    from math import gcd
    
    # Represent each reciprocal as a fraction
    # 1/a = (1, a)
    # We'll compute all subset sums and check for duplicates
    # For efficiency with n up to ~12, 2^12 = 4096 is fine
    
    # Use a list to build subset sums incrementally
    sums = [(0, 1)]  # (num, den) for empty set
    for a in S:
        new_sums = []
        for (num, den) in sums:
            # Current sum + 1/a
            # num/den + 1/a = (num*a + den) / (den*a)
            new_num = num * a + den
            new_den = den * a
            g = gcd(new_num, new_den)
            new_sums.append((new_num // g, new_den // g))
        sums.extend(new_sums)
    
    # Check for duplicates
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Universe: 1..22
    universe = list(range(1, 23))
    
    # Greedy: try to build a large dissociated set
    # Order: prefer smaller numbers? Or try multiple orderings
    
    best = []
    
    # Strategy: greedy with multiple orderings, then local search
    # Try adding elements in different orders
    
    def greedy_add(order):
        """Try to add elements in the given order, skipping conflicts."""
        current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            if _is_dissociated(test):
                current = test
        return current
    
    # Try several orderings
    import random
    random.seed(42)
    
    orderings = []
    # Natural order
    orderings.append(universe)
    # Reverse
    orderings.append(universe[::-1])
    # Random shuffles
    for _ in range(20):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_add(order)
        if len(result) > len(best):
            best = result
    
    # Local search: try to add one more element by removing a conflicting one
    # Or try swapping
    
    # Branch and bound for better results
    def branch_and_bound(available, current, depth=0):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) + len(available) <= len(best):
            return
        if not available:
            return
        
        # Try each available element
        for i, a in enumerate(available):
            if time.time() > deadline:
                return
            test = current + [a]
            if _is_dissociated(test):
                if len(test) > len(best):
                    best = test[:]
                remaining = available[:i] + available[i+1:]
                branch_and_bound(remaining, test, depth + 1)
            else:
                # Try without a
                pass
        
        # Also try not adding any more (already handled by len check)
    
    # Start B&B from the best greedy result
    # Actually, let's do a smarter B&B: pick elements to include/exclude
    
    # Better: iterative deepening or direct B&B from scratch
    # Since universe is only 22, and we expect answer ~10-12, B&B should work
    
    # Let's do a proper B&B
    def bb_search(available, current):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) + len(available) <= len(best):
            return
        if not available:
            if len(current) > len(best):
                best = current[:]
            return
        
        a = available[0]
        rest = available[1:]
        
        # Branch 1: include a
        if _is_dissociated(current + [a]):
            bb_search(rest, current + [a])
        
        # Branch 2: exclude a
        bb_search(rest, current)
    
    # To make B&B effective, order the universe so that promising elements come first
    # Smaller numbers create more conflicts, so maybe put larger numbers first?
    # Actually, let's try the current best as a starting point and see if we can extend
    
    # Let's just run B&B with a good ordering
    # Put larger numbers first (they have fewer conflicts)
    bb_order = sorted(universe, reverse=True)
    bb_search(bb_order, [])
    
    # Also try with the greedy result as a seed for further search
    # Remove elements from best and try to add different ones
    if len(best) >= 2:
        # Try removing one element and adding two others
        for i in range(len(best)):
            if time.time() > deadline:
                break
            reduced = best[:i] + best[i+1:]
            # Try to add any two elements not in reduced
            candidates = [x for x in universe if x not in reduced]
            for c1 in candidates:
                if time.time() > deadline:
                    break
                test1 = reduced + [c1]
                if _is_dissociated(test1):
                    for c2 in candidates:
                        if c2 == c1:
                            continue
                        if time.time() > deadline:
                            break
                        test2 = test1 + [c2]
                        if _is_dissociated(test2):
                            if len(test2) > len(best):
                                best = test2[:]
    
    return best
