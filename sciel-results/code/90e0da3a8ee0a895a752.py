# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def _is_valid_set(S):
    """Check if no element is the mean of any subset of at least 2 other elements."""
    n = len(S)
    if n < 3:
        return True
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        m = len(others)
        # We need to check if there exists a subset T of others with |T|>=2
        # such that sum(T) = len(T) * a
        # Equivalently, sum(T - a) = 0 where each element in T is (x - a)
        # Let d = [x - a for x in others]. We need a subset of d of size >=2 summing to 0.
        
        # Use bitset DP for subset sums? No, we need to track cardinality too, or just check all sizes.
        # Since n <= 512, m <= 511.
        # The values x - a can be negative.
        # Range of sums: min sum is sum of most negative, max sum is sum of most positive.
        # This can be large.
        
        # Alternative: Meet in the middle?
        # Or simple DP with dictionary of (sum, count) -> True?
        # The number of possible sums might be large.
        
        # Let's use a set of reachable sums for each cardinality k.
        # reachable[k] is a set of sums achievable by subsets of 'others' of size k.
        # We want to know if 0 is in reachable[k] for any k >= 2.
        
        # Optimization: If m is small, this is fast. If m is large, it might be slow.
        # But we are building a set of size ~48, so m ~ 47.
        # 2^47 is too large for brute force, but DP with sets is O(m * range).
        # Range of sums: max diff is 512, min is -512. Sum range is approx [-47*512, 47*512] ~ [-24000, 24000].
        # This is small enough for a bitset or boolean array DP.
        
        # Let's use a list of sets or a boolean array.
        # Since we only care about sum=0, and k>=2.
        
        # Actually, a simpler check:
        # For each a, check if there is a subset of others summing to k*a for k in 2..m.
        # This is equivalent to: is there a subset of others with average a?
        
        # Let's try a randomized check first for speed, or a direct DP if m is small.
        # For m=47, DP with bitsets is very fast.
        
        # Shift sums to be positive.
        # min_sum = sum(min(others, key=lambda x: x-a)[:k])? No, just bound the range.
        # Max absolute sum is sum(abs(x-a) for x in others).
        # Let's just use a set of reachable sums for each k, but prune if it gets too big?
        # Or use numpy?
        
        # Let's use a standard DP with a dictionary of sums for each k, but only keep sums that are "plausible" to reach 0?
        # Actually, since the target is 0, we can just track reachable sums.
        
        # For performance, we'll use a list of sets.
        # dp[k] = set of sums achievable with k elements.
        
        # Base case: dp[0] = {0}
        # But we need k>=2.
        
        # Let's implement a helper to check if 0 is reachable with k>=2 elements from a list of diffs.
        def _zero_subset_exists(diffs):
            m = len(diffs)
            if m < 2:
                return False
            # dp[k] is a set of sums
            # To save memory/time, we can use a boolean array if the range is small.
            # Max sum magnitude: sum(abs(d) for d in diffs).
            # Let's estimate max_range.
            max_range = sum(abs(d) for d in diffs)
            if max_range > 100000: # Fallback to something else? Unlikely for n=512.
                # Use set-based DP
                dp = [set() for _ in range(m+1)]
                dp[0].add(0)
                for d in diffs:
                    for k in range(m-1, -1, -1):
                        if dp[k]:
                            for s in list(dp[k]):
                                dp[k+1].add(s+d)
                for k in range(2, m+1):
                    if 0 in dp[k]:
                        return True
                return False
            else:
                # Bitset DP
                # offset = max_range
                # size = 2*max_range + 1
                # This might be too big for numpy if max_range is large, but 100k is fine.
                # Actually, let's just use sets for simplicity and correctness, it's fast enough for m~50.
                dp = [set() for _ in range(m+1)]
                dp[0].add(0)
                for d in diffs:
                    for k in range(m-1, -1, -1):
                        if dp[k]:
                            # Optimization: if 0 is already in dp[k+1], we can return early? No, we are building it.
                            for s in dp[k]:
                                dp[k+1].add(s+d)
                for k in range(2, m+1):
                    if 0 in dp[k]:
                        return True
                return False

        diffs = [x - a for x in others]
        if _zero_subset_exists(diffs):
            return False
    return True

def _check_candidate(S, cand):
    """Check if adding cand to S keeps the set valid.
    We only need to check:
    1. Is cand the mean of a subset of S?
    2. For each a in S, is a the mean of a subset of S U {cand}?
       This means either a was already the mean of a subset of S (which is false by invariant),
       or a is the mean of a subset of S that includes cand.
       So we just need to check if a is the mean of a subset T of S such that (sum(T) + cand) / (len(T)+1) == a
       => sum(T) + cand = a(len(T)+1)
       => sum(T) - a*len(T) = a - cand
       Let d_T = [x - a for x in T]. We need sum(d_T) = a - cand.
    """
    # 1. Check cand
    others = list(S)
    m = len(others)
    if m >= 2:
        diffs = [x - cand for x in others]
        # Check if 0 is in subset sums of diffs with size >= 2
        # Reuse logic
        def _zero_subset_exists_fast(diffs):
            m = len(diffs)
            if m < 2: return False
            dp = [set() for _ in range(m+1)]
            dp[0].add(0)
            for d in diffs:
                for k in range(m-1, -1, -1):
                    if dp[k]:
                        for s in dp[k]:
                            dp[k+1].add(s+d)
            for k in range(2, m+1):
                if 0 in dp[k]:
                    return True
            return False
        if _zero_subset_exists_fast(diffs):
            return False

    # 2. Check each a in S
    for a in S:
        target = a - cand
        # We need a subset T of S (others of a, so S without a) such that sum(x - a for x in T) = target
        # Let U = S without a.
        U = [x for x in S if x != a]
        diffs = [x - a for x in U]
        # Find if target is in subset sums of diffs.
        # Size of T can be anything (0 to len(U)).
        # Note: The subset in the original problem must have size >= 2 OTHER elements.
        # The subset is T U {cand}. Size is len(T)+1.
        # We need len(T)+1 >= 2 => len(T) >= 1.
        # So T must be non-empty.
        
        # DP for subset sums
        m = len(diffs)
        if m == 0:
            continue
            
        # We want to know if target is achievable by a non-empty subset.
        # dp[s] = min number of elements used to achieve sum s? Or just reachable?
        # We just need reachability.
        # Since we need non-empty, we can just check if target is reachable and not (target==0 and only empty set).
        # Actually, if target != 0, any subset achieving it is non-empty.
        # If target == 0, we need a non-empty subset summing to 0.
        
        # Let's use a set of reachable sums.
        reachable = {0}
        # To ensure non-empty for target=0, we can track if 0 is reachable by non-empty subset.
        # Or simpler: just find any subset summing to target. If target==0, exclude empty set.
        
        # For efficiency, we can stop early if found.
        # But we need to be careful with the non-empty constraint for target=0.
        
        # Let's use a dict: sum -> min_size? No, just set of sums.
        # And a separate flag for empty set?
        # Actually, if target != 0, empty set (sum 0) doesn't matter.
        # If target == 0, we need a non-empty subset.
        
        # Let's just compute all reachable sums.
        # m is at most 47.
        # Number of sums can be large, but usually bounded.
        
        sums = {0}
        for d in diffs:
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + d)
            sums = new_sums
            
        if target in sums:
            if target != 0:
                return False
            else:
                # Check if 0 is achievable by non-empty subset
                # We can check if there is any d==0 (subset of size 1)
                # Or any pair d1+d2=0, etc.
                # Simpler: re-run DP tracking size?
                # Or just check if any single diff is 0.
                # Or check if any subset of size >=1 sums to 0.
                # Since 0 is in sums, it's either empty or non-empty.
                # If any d is 0, then size 1 works.
                if 0 in diffs:
                    return False
                # Else, check if any subset of size >=2 sums to 0.
                # This is the same as _zero_subset_exists but we already know 0 is reachable.
                # We can just return False if we can find a non-empty subset.
                # For now, let's assume if 0 is in sums and not all diffs are non-zero in a way that only empty works...
                # Actually, if 0 is in sums, and no single diff is 0, it must be a subset of size >=2.
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    # Greedy with randomization and local search
    # Strategy:
    # 1. Generate multiple candidate sets using different heuristics.
    # 2. Try to extend them.
    # 3. Local search: swap elements to fix conflicts or add more.
    
    best_set = []
    
    # Initial seed: try to build a large set quickly
    # Use a greedy approach with random ordering
    
    def greedy_build(order, max_time=None):
        S = []
        for x in order:
            if time.time() > (max_time or deadline):
                break
            if _check_candidate(S, x):
                S.append(x)
        return S
    
    # Try several random orders
    for i in range(20):
        if time.time() > deadline - 5.0:
            break
        # Random permutation of 1..512
        order = list(range(1, 513))
        random.shuffle(order)
        
        # Bias towards numbers that are "spread out" or have specific properties?
        # For now, just random.
        
        S = greedy_build(order)
        
        if len(S) > len(best_set):
            best_set = S
            
        # Local search: try to replace elements to allow more additions
        # This is complex. Let's stick to multiple greedy runs first.
        
    # Try to extend best_set by trying all remaining numbers
    if best_set:
        S_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in S_set]
        for x in remaining:
            if time.time() > deadline - 2.0:
                break
            if _check_candidate(best_set, x):
                best_set.append(x)
                S_set.add(x)
                
    return best_set
# EVOLVE-BLOCK-END
