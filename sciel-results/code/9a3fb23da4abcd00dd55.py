import random
import time
from itertools import combinations

def construct():
    # Elements are 0..15
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    # Using itertools.combinations
    candidates = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
    
    # Shuffle candidates initially? No, we shuffle in each restart.
    
    best_family = []
    best_size = 0
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin
    
    # We can also try to order candidates heuristically.
    # But random restarts are robust.
    
    # Optimization: Pre-compute intersections? No, too much memory.
    
    # Helper to check if s can be added to family
    def can_add(s, family):
        m = len(family)
        # Check all pairs in family
        for i in range(m):
            a = family[i]
            for j in range(i + 1, m):
                b = family[j]
                # Check if a, b, s form a sunflower
                # Intersections: a&b, a&s, b&s must be equal
                # a&b == a&s == b&s
                # This is equivalent to:
                # (a & b) == (a & s) and (a & b) == (b & s)
                # Or simply:
                if (a & b) == (a & s) == (b & s):
                    return False
        return True

    while time.time() - start_time < time_limit:
        # Shuffle candidates
        random.shuffle(candidates)
        
        family = []
        for s in candidates:
            # Check if s can be added
            # Optimization: If family is large, check might be slow.
            # But M is small.
            if can_add(s, family):
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            # Early termination if we reach the theoretical max or high bound?
            # The task says max 250. If we hit 250, we can stop?
            # "1..maximum=250" suggests the output size is bounded by 250, 
            # or the search space is bounded. 
            # Actually, it says "Maximize family cardinality". 
            # If we find 250, is it the max? 
            # The prompt says "1..maximum=250 distinct k=4 element subsets".
            # This phrasing is slightly ambiguous. It might mean the output list 
            # should have length between 1 and 250. 
            # Or it might mean the maximum possible size is 250.
            # Given "Maximize family cardinality", we should aim for the max.
            # If we hit 250, we can probably stop searching for larger, 
            # but we should ensure validity.
            if best_size >= 250:
                break

    # Convert bitmasks back to sorted lists of elements
    # The evaluator expects distinct k=4 element subsets.
    # Format: list of lists, each inner list sorted.
    
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1) # 1-indexed
        result.append(elems)
        
    return result
