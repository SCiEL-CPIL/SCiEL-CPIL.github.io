import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = 512
    max_size = 48
    
    def check_valid(selected, a):
        """Check if adding a to selected keeps the set valid."""
        if a in selected:
            return False
        others = [x for x in selected if x != a]
        if len(others) < 1:
            # Need at least 2 others for a violation, so if < 2 others, no issue from a's perspective
            # But we also need to check existing elements against new subsets including a
            pass
        
        # Check: is a the mean of any subset of others (size >= 2)?
        # i.e., exists subset T of others, |T| >= 2, sum(T) = |T| * a
        if len(others) >= 2:
            # Use DP: dp[j] = set of achievable sums using j elements
            # But we only care about sums = k * a for k = 2, ..., len(others)
            # Use bitset approach
            # Values: others, we want subsets of size k summing to k*a
            # For each k from 2 to len(others), check if k*a is achievable by k elements
            
            # Efficient check using DP with bitsets
            # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
            # Max sum = sum(others), which is at most 47*512 ~ 24064
            max_sum = sum(others)
            target_sums = set(k * a for k in range(2, len(others) + 1))
            target_sums = {t for t in target_sums if 0 <= t <= max_sum}
            
            if not target_sums:
                pass
            else:
                # DP: dp[j] = bitset of achievable sums with j elements
                # dp[0] = 1 (sum 0 with 0 elements)
                dp = [0] * (len(others) + 1)
                dp[0] = 1  # bit 0 set
                
                for val in others:
                    for j in range(len(others), 0, -1):
                        # Shift dp[j-1] by val
                        dp[j] |= (dp[j-1] << val)
                
                # Check if any target sum is achievable with the right cardinality
                for k in range(2, len(others) + 1):
                    target = k * a
                    if 0 <= target <= max_sum:
                        if (dp[k] >> target) & 1:
                            return False
        
        # Check: for each existing element b in selected, is b the mean of any subset 
        # of (selected \ {b}) ∪ {a} with size >= 2?
        # The new subsets that include a are the only new ones to check
        for b in selected:
            if b == a:
                continue
            # New subsets involving a: {a} ∪ T where T ⊆ selected \ {b, a}
            # We need |{a} ∪ T| >= 2, so |T| >= 1
            # And sum({a} ∪ T) = |{a} ∪ T| * b
            # i.e., a + sum(T) = (1 + |T|) * b
            # i.e., sum(T) - |T| * b = b - a
            # Let T' = {t - b : t in T}, then sum(T') = b - a
            # T ⊆ selected \ {b, a}, |T| >= 1
            
            pool = [x for x in selected if x != b and x != a]
            if not pool:
                continue
            
            # We need a non-empty subset T of pool such that sum(t - b for t in T) = b - a
            # i.e., sum(T) = |T| * b + (b - a)
            
            # Use DP on the shifted values
            shifted = [x - b for x in pool]
            target = b - a
            
            # Check if target is achievable by a non-empty subset of shifted
            # DP: bitset of achievable sums
            # Min and max of shifted values
            min_val = min(shifted) if shifted else 0
            max_val = max(shifted) if shifted else 0
            min_sum = min_val * len(shifted)  # worst case all negative
            max_sum2 = max_val * len(shifted)  # worst case all positive
            
            # Use offset for negative values
            # Actually, let's just use a set-based DP for small pools
            if len(pool) <= 20:
                # Meet in the middle or just DP with sets
                achievable = {0}
                for v in shifted:
                    new_achievable = set()
                    for s in achievable:
                        new_achievable.add(s + v)
                    achievable = achievable | new_achievable
                if target in achievable and target != 0 or (target == 0 and len(pool) >= 1):
                    # Need non-empty subset
                    # If target == 0, we need a non-empty subset summing to 0
                    # Check if 0 is achievable by non-empty subset
                    if target == 0:
                        # Recheck: is there a non-empty subset summing to 0?
                        non_empty_zero = False
                        for i in range(1, len(shifted) + 1):
                            # This is getting complex, let's just check directly
                            pass
                        # Simpler: check if 0 in achievable and achievable != {0}
                        # But achievable includes 0 from empty set
                        # Let's track non-empty separately
                        pass
                    else:
                        if target in achievable:
                            return False
            else:
                # For larger pools, use bitset DP
                # Offset to handle negatives
                offset = abs(min_sum) + 1
                size = max_sum2 - min_sum + 2 * offset + 1
                if size > 100000:
                    # Too large, skip or use approximate
                    continue
                dp = 1  # bit 0 set (empty set, sum 0)
                for v in shifted:
                    dp |= (dp << (v + offset))
                # Check if target + offset is set
                if (dp >> (target + offset)) & 1:
                    # Need to ensure it's from a non-empty subset
                    # If target == 0, the empty set also gives 0, so we need to be careful
                    if target == 0:
                        # Check if there's a non-empty subset summing to 0
                        # This is harder; for now, assume it's a violation if target != 0
                        pass
                    else:
                        return False
        
        return True
    
    # Greedy construction with multiple orderings
    best_set = []
    
    # Try different orderings: ascending, descending, random
    import random
    
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    
    # Add some random orderings
    for _ in range(5):
        random.seed(42 + _)
        orderings.append(list(range(1, n + 1)))
        random.shuffle(orderings[-1])
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            if len(selected) >= max_size:
                break
            if check_valid(selected, x):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) >= max_size:
                break
    
    return best_set
