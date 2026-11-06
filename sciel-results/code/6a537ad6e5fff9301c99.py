"""Search for optimal pairwise-coprime sumset construction."""
import itertools
import math
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute gcd
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Check if all values in a list are pairwise coprime
    def all_pairwise_coprime(vals):
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    best_count = 0
    best_A = None
    best_B = None
    
    # All possible 3-element sets from [1, 128]
    # That's C(128,3) = 341,376 possibilities
    # Pairwise: 341376^2 is too large for full search
    # Strategy: iterate over A, for each A iterate over B, check validity
    
    # Precompute all valid 3-element sets
    all_sets = list(itertools.combinations(range(1, 129), 3))
    n_sets = len(all_sets)
    
    # For speed, precompute the sumset for each set pair is expensive
    # Instead: iterate A, iterate B, compute sums, check pairwise coprime
    # 341376 * 341376 is ~1.17e11, too slow
    # Need smarter approach
    
    # Alternative: since we want to maximize distinct sum count (max 9),
    # and the sumset has at most 9 distinct values, we can:
    # 1. First try to find solutions with 9 distinct pairwise-coprime sums
    # 2. If not found, try 8, etc.
    
    # Key insight: for 9 distinct sums, all 9 must be pairwise coprime
    # The 9 sums are a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # where A = {a1,a2,a3}, B = {b1,b2,b3}
    
    # For 9 distinct sums to be pairwise coprime, they must be 9 distinct integers
    # that are pairwise coprime. Small pairwise coprime sets: 1,2,3,5,7,11,13,17,19
    # but 1 is coprime to everything.
    
    # Let me try a more targeted search:
    # For each A (3-element set), compute what B would need to satisfy
    # Actually, let's just do the full search but with early termination
    
    # Since 341376^2 is too large, let's be smarter:
    # For a given A, the sumset A+B has elements a_i + b_j.
    # We need all distinct sums to be pairwise coprime.
    
    # Let me try: iterate over all A, and for each A, iterate over all B
    # but break early if we find 9 distinct pairwise-coprime sums
    
    # Actually 341376 * 341376 = 1.165e11 - too slow in Python
    # Let's reduce: note that sums range from 1+1=2 to 128+128=256
    # So distinct sums are in [2, 256]. For pairwise coprime in this range...
    
    # Better approach: enumerate candidate sumsets first, then check realizability
    # Or: use the structure. Let A = {a,b,c} with a<b<c, B = {d,e,f} with d<e<f
    # Sums: a+d, a+e, a+f, b+d, b+e, b+f, c+d, c+e, c+f
    # For 9 distinct: need a+d != a+e != ... all different, which means 
    # a+d < a+e < a+f < b+d < ... requires specific ordering
    
    # For 9 distinct sums, we need: a+f < b+d (otherwise a+f >= b+d means not all distinct in a grid)
    # Actually the sums form a 3x3 grid. For all 9 to be distinct:
    # a+d, a+e, a+f (row 1)
    # b+d, b+e, b+f (row 2)  
    # c+d, c+e, c+f (row 3)
    # All distinct iff the differences b-a, c-a, e-d, f-d are such that no two sums collide.
    # a+e = b+d iff e-d = b-a. So we need e-d != b-a and f-d != b-a and f-d != c-a and e-d != c-a.
    
    # This is getting complex. Let me just do a randomized + structured search.
    
    # Phase 1: Try known good structures
    # The seed [[1,7,13],[4,10,16]] gives sums: 5,11,17,11,17,23,17,23,29 -> distinct: {5,11,17,23,29} = 5
    
    # Let me try to find 6, 7, 8, or 9 distinct pairwise-coprime sums.
    
    # For 9 distinct pairwise coprime sums in [2,256]:
    # Need 9 numbers in [2,256] that are pairwise coprime.
    # Example: 2,3,5,7,11,13,17,19,23 (all primes) - these are pairwise coprime!
    # Can we realize {2,3,5,7,11,13,17,19,23} as A+B with |A|=|B|=3?
    # The sumset of two 3-sets forms a "grid" pattern.
    
    # Let me try a direct search with time limit:
    # Iterate A from all 3-sets, for each try several B's
    
    # Smarter: fix the sumset structure. If we want 9 distinct sums,
    # the 3x3 grid must have all distinct entries.
    # Let A = {0, x, y} (shift doesn't matter for differences, but values must be in [1,128])
    # Actually let's just do: for A in all_sets, for B in all_sets (with early exit)
    
    # Given time constraints, let me do a focused search:
    # Try all A, and for each A, sample many B's, check validity
    
    import random
    random.seed(42)
    
    # Precompute: for each 3-set, store as tuple for speed
    all_sets_tup = [tuple(s) for s in all_sets]
    
    # First, try to find 9 distinct pairwise-coprime sums
    # The sums are in [2, 256]. Let's check which 9-element subsets of [2,256] are pairwise coprime
    # and could be a sumset of two 3-sets.
    
    # A sumset of two 3-sets {a1<a2<a3} and {b1<b2<b3} has the property that:
    # min sum = a1+b1, max sum = a3+b3
    # The 9 sums include a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # For all 9 to be distinct and pairwise coprime:
    # Let me just do a brute force with the constraint that we check pairwise coprime early
    
    # Given the time budget, let's do:
    # For each A (341376 options), compute the "differences" pattern
    # Then for B, the sums are determined
    
    # Actually, let me just iterate and hope we find something good within time.
    # 341376 * some_fraction_of_B's
    
    # Let me try: for each A, try all B's but with vectorized-ish checks
    # This is 341376^2 = 1.17e11 which is way too slow.
    
    # Alternative: random search
    best = 0
    best_pair = ([1,7,13], [4,10,16])
    
    # Try random pairs
    iterations = 0
    for _ in range(5000000):  # up to 5M random trials
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        A.sort()
        B.sort()
        
        # Compute sumset
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if len(sums) <= best:
            continue
        
        # Check pairwise coprime
        vals = sorted(sums)
        ok = True
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            best = len(sums)
            best_pair = (A, B)
            if best == 9:
                break
    
    # Also try structured search around the seed
    # Try small sets
    small_range = range(1, 50)
    for A in itertools.combinations(small_range, 3):
        if time.time() > deadline:
            break
        for B in itertools.combinations(small_range, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) <= best:
                continue
            
            vals = sorted(sums)
            ok = True
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    if math.gcd(vals[i], vals[j]) != 1:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                best = len(sums)
                best_pair = (list(A), list(B))
                if best == 9:
                    return [best_pair[0], best_pair[1]]
    
    if time.time() > deadline - 5:
        return list(best_pair)
    
    # Try slightly larger structured search
    mid_range = range(1, 80)
    for A in itertools.combinations(mid_range, 3):
        if time.time() > deadline - 3:
            break
        for B in itertools.combinations(mid_range, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) <= best:
                continue
            
            vals = sorted(sums)
            ok = True
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    if math.gcd(vals[i], vals[j]) != 1:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                best = len(sums)
                best_pair = (list(A), list(B))
                if best == 9:
                    return [best_pair[0], best_pair[1]]
    
    return [list(best_pair[0]), list(best_pair[1])]
