import time
import itertools

def construct():
    start_time = time.time()
    deadline = 90.0  # Safety margin
    
    # The problem is to find a large subset S of {1, ..., 24}
    # such that for all x in S, 1/x is not the sum of reciprocals of any subset of S \ {x}.
    
    # We suspect large numbers are safe.
    # Let's try to build the set greedily from largest to smallest,
    # checking validity at each step.
    
    # Validity check:
    # For a candidate set S, for each x in S,
    # check if 1/x can be formed by a subset of S \ {x}.
    # Since |S| <= 24, we can use bitmasks or simple recursion.
    # But checking all subsets for all x is O(N * 2^N).
    # N=24, 2^24 is 16 million. Too slow for repeated checks in search.
    # However, we can precompute sums of reciprocals for all subsets?
    # Or just check incrementally.
    
    # A faster check:
    # For a fixed x, we want to know if 1/x is in the set of sums of reciprocals of S \ {x}.
    # We can use a set of reachable sums.
    
    def is_valid_set(S):
        S_set = set(S)
        for x in S:
            others = [y for y in S if y != x]
            target = 1.0 / x
            # Check if target can be formed by subset of others
            # Use a set of sums to avoid duplicates and prune
            sums = {0.0}
            for y in others:
                new_sums = set()
                for s in sums:
                    ns = s + 1.0 / y
                    if abs(ns - target) < 1e-12:
                        return False
                    if ns < target:
                        new_sums.add(ns)
                sums.update(new_sums)
                # Optimization: if min(sums) > target, break? No, sums are positive.
                # If all sums > target, we can stop?
                # Actually, if the smallest sum > target, we can't reach target.
                # But we are adding positive numbers.
                # If min(sums) > target, then all future sums will be > target.
                if min(sums) > target:
                    break
            # If we finish the loop without finding target, x is safe.
        return True

    # Greedy construction
    # Start with empty set, add elements from 24 down to 1.
    # If adding x keeps the set valid, add it.
    
    S = []
    for x in range(24, 0, -1):
        candidate = S + [x]
        if is_valid_set(candidate):
            S.append(x)
            
    # Local search: try to add more?
    # The greedy approach might miss combinations.
    # But for this specific "reciprocal subsum-free" property, large numbers are very "sparse" in terms of reciprocal sums.
    # The greedy approach from large to small is likely optimal or near-optimal.
    
    # Let's also try to remove small elements if they cause issues?
    # The greedy check ensures validity.
    
    # Check time
    if time.time() > deadline - 5:
        return S
        
    # Try to improve with local search:
    # 1. Try to add an element that was rejected.
    # 2. If it fails, try to remove another element to make room?
    
    # Given the time limit, a simple greedy is probably sufficient to beat 12.
    # The greedy will likely find 16-18 elements.
    
    return S
