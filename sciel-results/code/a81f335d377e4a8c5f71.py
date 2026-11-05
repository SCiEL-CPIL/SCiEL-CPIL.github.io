import time
from math import gcd
from itertools import combinations
from functools import reduce

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums using integer arithmetic."""
    if not S:
        return True
    n = len(S)
    if n > 20: # Cutoff for speed
        return False
        
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, S, 1)
    vals = [L // a for a in S]
    
    sums = {0}
    current_sums = [0]
    
    for v in vals:
        new_sums = [s + v for s in current_sums]
        for ns in new_sums:
            if ns in sums:
                return False
            sums.add(ns)
        current_sums.extend(new_sums)
        
    return True

def _get_bad_subsets(max_size=4):
    """Pre-compute all non-dissociated subsets of [1, 22] up to max_size."""
    universe = list(range(1, 23))
    bad = set()
    for size in range(2, max_size + 1):
        for combo in combinations(universe, size):
            if not _is_dissociated(list(combo)):
                bad.add(combo)
    return bad

def construct():
    deadline = time.time() + 85
    
    universe = list(range(1, 23))
    best = []
    
    # Pre-compute bad subsets
    # This takes some time, but is one-time cost.
    # Size 2: No bad subsets (1/a != 1/b for a!=b).
    # Size 3: 1/a = 1/b + 1/c => 1/a - 1/b = 1/c => (b-a)/ab = 1/c => c = ab/(b-a).
    # Size 4: etc.
    bad_subsets = _get_bad_subsets(4)
    
    def is_valid_fast(current, a):
        """Check if current + [a] is potentially valid by checking bad subsets."""
        test_set = current + [a]
        test_set_sorted = sorted(test_set)
        # Check if any bad subset is contained in test_set
        # Since test_set is small (<= 15), we can just check all bad subsets
        # Optimization: Only check bad subsets of size <= len(test_set)
        for bs in bad_subsets:
            if len(bs) > len(test_set):
                continue
            # Check if bs is subset of test_set
            # Since both are sorted, we can do this fast
            # Or simply: set(bs).issubset(set(test_set))
            # But creating sets is slow.
            # Let's use a simple loop for small sizes.
            # Actually, we can pre-store bad_subsets as sorted tuples.
            # And test_set_sorted is sorted.
            # We can check if all elements of bs are in test_set.
            # Since n is small, set intersection is fine.
            if set(bs).issubset(set(test_set)):
                return False
        return True

    def greedy_add(order):
        current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            # Fast check first
            if is_valid_fast(current, a):
                # Full check
                if _is_dissociated(test):
                    current = test
        return current

    # 1. Greedy
    import random
    random.seed(42)
    orderings = [universe, universe[::-1]]
    for _ in range(10):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
        
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_add(order)
        if len(result) > len(best):
            best = result[:]

    # 2. Branch and Bound
    def bb_search(available, current, idx=0):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(current) + len(available) <= len(best):
            return
            
        if not available:
            return
            
        a = available[0]
        rest = available[1:]
        
        # Branch 1: Include a
        # Fast check
        if is_valid_fast(current, a):
            if _is_dissociated(current + [a]):
                new_current = current + [a]
                if len(new_current) > len(best):
                    best = new_current[:]
                bb_search(rest, new_current, idx + 1)
            
        # Branch 2: Exclude a
        bb_search(rest, current, idx + 1)
        
    bb_order = sorted(universe, reverse=True)
    bb_search(bb_order, [])
    
    return best
