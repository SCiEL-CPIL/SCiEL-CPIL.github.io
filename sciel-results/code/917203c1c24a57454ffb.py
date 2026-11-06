"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_dissociated_recip(elements):
    """Check if all reciprocal subset sums are distinct."""
    n = len(elements)
    if n == 0:
        return True
    # Use scaled integers: multiply by LCM to avoid floating point
    from math import gcd
    from functools import reduce
    
    # Compute LCM of all elements
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    try:
        l = reduce(lcm, elements)
    except Exception:
        return False
    
    # Scale each element: weight_i = l // elements[i]
    weights = [l // e for e in elements]
    
    # Check all 2^n subset sums are distinct
    # For efficiency, use a set
    sums = set()
    n = len(weights)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += weights[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _is_dissociated_fast(elements):
    """Faster check using incremental subset sum generation."""
    n = len(elements)
    if n == 0:
        return True
    from math import gcd
    from functools import reduce
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    try:
        l = reduce(lcm, elements)
    except Exception:
        return False
    
    weights = [l // e for e in elements]
    
    # Generate all subset sums incrementally
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
        if len(sums) != (1 << (len(sums) - len(new_sums))):  # sanity
            pass
    return True

def _check_validity(candidate):
    """Verify candidate is a valid nonempty subset of [1,22] with distinct reciprocal subset sums."""
    if not candidate:
        return False
    if len(candidate) != len(set(candidate)):
        return False
    if any(x < 1 or x > 22 for x in candidate):
        return False
    return _is_dissociated_fast(sorted(candidate))

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    # Start with a known good set
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try to find a larger set by adding elements
    # Strategy: try all subsets of increasing size, check dissociation
    
    # Precompute which elements can potentially be added
    all_elements = list(range(1, 23))
    
    # Try sizes from 10 up to 22
    for size in range(10, 23):
        if time.time() > deadline:
            break
        
        # Try combinations of this size
        # For efficiency, try greedy: start from best and try to add
        # First, try all combinations of size 'size'
        # But that's too many for size > 12, so use heuristic
        
        if size <= 12:
            for combo in combinations(all_elements, size):
                if time.time() > deadline:
                    break
                if _is_dissociated_fast(list(combo)):
                    best = list(combo)
                    break  # Found one of this size, try larger
        else:
            # For larger sizes, use a greedy approach
            # Start with a large set and try to verify
            # Heuristic: try sets that are "well-spread"
            candidates_to_try = []
            
            # Try the full set and subsets
            # A key insight: if 1 is in the set, then for any other element x,
            # we need 1/x not to equal any sum of reciprocals of other elements
            # This is very restrictive
            
            # Let's try a different approach: use the fact that powers of 2
            # scaled appropriately work. But we're limited to [1,22].
            
            # Try specific constructions
            # One approach: use elements that are "superincreasing" in their reciprocals
            # i.e., 1/a_i > sum of 1/a_j for all j > i
            # This guarantees dissociation
            
            # Let's try to find the largest superincreasing reciprocal set
            # Sort elements in decreasing order of value (increasing order of reciprocal)
            # Actually, we want 1/a_1 > 1/a_2 > ... > 1/a_k
            # and 1/a_i > sum_{j>i} 1/a_j
            
            # Greedy: start with the smallest element (largest reciprocal)
            # and add the next smallest that satisfies the condition
            
            # Try greedy construction
            for start_elem in range(1, 23):
                if time.time() > deadline:
                    break
                current = [start_elem]
                for candidate in range(start_elem + 1, 23):
                    if time.time() > deadline:
                        break
                    # Check if adding candidate preserves superincreasing property
                    # Need: 1/current[-1] > 1/candidate + sum of all future... 
                    # Actually, the condition is: 1/candidate < sum of reciprocals of all smaller elements in current
                    # Wait, let me reconsider.
                    
                    # For superincreasing: if we have elements a1 < a2 < ... < ak (so 1/a1 > 1/a2 > ... > 1/ak)
                    # Then we need 1/ai > sum_{j>i} 1/aj for all i
                    # This is sufficient but not necessary for dissociation
                    
                    # Let's just check if the set is dissociated
                    test_set = current + [candidate]
                    if _is_dissociated_fast(test_set):
                        current = test_set
                
                if len(current) > len(best):
                    best = current
                    break  # Found a better one with this start
            
            # Also try some hand-crafted large sets
            handcrafted = [
                [1, 2, 4, 8, 16, 22],  # powers of 2 plus 22
                [1, 2, 4, 8, 16, 20, 22],
                [1, 2, 3, 6, 12, 24],  # but 24 > 22
                [1, 2, 3, 6, 12, 22],
                [1, 2, 3, 5, 7, 11, 13, 17, 19, 21],
                [1, 2, 3, 5, 7, 11, 13, 17, 19, 20],
                [1, 2, 3, 5, 7, 11, 13, 17, 19, 22],
                [1, 2, 4, 8, 16, 20, 22],
                [1, 2, 4, 8, 16, 19, 22],
                [1, 2, 4, 8, 15, 20, 22],
                [1, 2, 3, 6, 12, 18, 22],
                [1, 2, 3, 6, 12, 20, 22],
                [1, 2, 3, 6, 12, 21, 22],
                [1, 2, 3, 5, 10, 20, 22],
                [1, 2, 3, 5, 10, 21, 22],
                [1, 2, 3, 5, 10, 15, 22],
                [1, 2, 3, 5, 10, 15, 20],
                [1, 2, 3, 5, 10, 15, 21],
            ]
            for hc in handcrafted:
                if time.time() > deadline:
                    break
                if len(hc) > len(best) and _is_dissociated_fast(hc):
                    best = hc
    
    # Final verification
    if not _check_validity(best):
        # Fallback to known good
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
# EVOLVE-BLOCK-END
