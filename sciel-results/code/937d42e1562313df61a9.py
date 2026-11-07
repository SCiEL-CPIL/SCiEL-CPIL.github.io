import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # Known valid 31-element solution: AP [226..256]
    fallback = list(range(226, 257))
    best_size = 31
    best_set = fallback
    
    # Quick validation using bitsets for speed
    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        # For each cardinality k, compute the range of possible sums
        # min_sum[k] = sum of k smallest elements
        # max_sum[k] = sum of k largest elements
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        
        # Check that for all k1 < k2, max_sum[k1] < min_sum[k2]
        # This is sufficient AND necessary for the condition
        for k1 in range(n + 1):
            max_sum_k1 = prefix[n] - prefix[n - k1]
            for k2 in range(k1 + 1, n + 1):
                min_sum_k2 = prefix[k2]
                if max_sum_k1 >= min_sum_k2:
                    return False
        return True
    
    # Bitset-based DP to check if a candidate set is valid
    # For a set S of size n, we track for each cardinality k which sums are achievable
    # Then check that sum ranges for different cardinalities are disjoint
    
    def check_with_bitset(s):
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        
        # dp[k] is a bitset where bit i is set if sum i is achievable with k elements
        # Maximum possible sum is sum of all elements
        total_sum = sum(s_sorted)
        
        # Use Python integers as bitsets
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in s_sorted:
            # Process in reverse to avoid using x multiple times
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << x
        
        # Now check: for each k, find min and max achievable sum
        # min_achievable[k] = lowest set bit in dp[k]
        # max_achievable[k] = highest set bit in dp[k]
        
        ranges = []
        for k in range(n + 1):
            if dp[k] == 0:
                continue
            # Find lowest set bit
            min_sum = (dp[k] & -dp[k]).bit_length() - 1
            # Find highest set bit
            max_sum = dp[k].bit_length() - 1
            ranges.append((k, min_sum, max_sum))
        
        # Check that ranges for different k are disjoint
        # Sort by min_sum and check no overlap
        ranges.sort(key=lambda r: r[1])
        for i in range(len(ranges) - 1):
            if ranges[i][2] >= ranges[i + 1][1]:
                return False
        return True
    
    # Fast check using prefix sums (necessary and sufficient condition)
    def fast_check(s):
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        
        # For the condition to hold, we need:
        # For all k1 < k2: max_sum(k1) < min_sum(k2)
        # max_sum(k) = sum of k largest = prefix[n] - prefix[n-k]
        # min_sum(k) = sum of k smallest = prefix[k]
        
        for k1 in range(n + 1):
            max_sum_k1 = prefix[n] - prefix[n - k1]
            for k2 in range(k1 + 1, n + 1):
                min_sum_k2 = prefix[k2]
                if max_sum_k1 >= min_sum_k2:
                    return False
        return True
    
    # Greedy construction with bitset validation
    def greedy_build(target_size, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Start with empty set and greedily add elements
        # Strategy: try to build from both ends (small and large elements)
        
        # First, try APs with various parameters
        for n in range(target_size, min(target_size + 5, 61)):
            if time.time() > deadline:
                break
            for d in range(1, 5):
                max_a = 256 - (n - 1) * d
                if max_a < 1:
                    continue
                # Try several starting points
                for a in range(1, max_a + 1, max(1, max_a // 20)):
                    s = [a + i * d for i in range(n)]
                    if s[-1] > 256:
                        continue
                    if fast_check(s):
                        return s
        return None
    
    # Try to find larger sets using various strategies
    for target in range(32, 45):
        if time.time() > deadline:
            break
        
        # Strategy 1: AP search
        result = greedy_build(target)
        if result and len(result) > best_size:
            best_size = len(result)
            best_set = result
            continue
        
        # Strategy 2: Two-cluster approach
        # Split elements into small and large clusters
        if time.time() > deadline:
            break
        
        for n_small in range(max(1, target // 4), min(target - 1, target * 3 // 4)):
            if time.time() > deadline:
                break
            n_large = target - n_small
            
            # Small cluster: consecutive starting from 1
            # Large cluster: consecutive ending at 256
            s_small = list(range(1, n_small + 1))
            s_large = list(range(257 - n_large, 257))
            
            # Check for overlap
            if s_small and s_large and s_small[-1] >= s_large[0]:
                continue
            
            s = s_small + s_large
            if len(set(s)) == target and fast_check(s):
                if target > best_size:
                    best_size = target
                    best_set = s
                break
        
        # Strategy 3: Randomized local search
        if time.time() > deadline - 5:
            break
        
        # Start from a base and try random modifications
        base = list(range(226, 257))  # 31 elements
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Randomly perturb the base set
            s = base[:]
            random.shuffle(s)
            
            # Try to add elements
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(s) >= target:
                    break
                # Try adding a random element
                candidates = [x for x in range(1, 257) if x not in s]
                if not candidates:
                    break
                x = random.choice(candidates)
                s.append(x)
                s.sort()
                if fast_check(s):
                    continue
                else:
                    s.remove(x)
            
            if len(s) > best_size and fast_check(s):
                best_size = len(s)
                best_set = s
    
    # Strategy 4: Systematic search for 32 elements
    if best_size < 32 and time.time() < deadline - 10:
        # Try all contiguous ranges of length 32
        for a in range(1, 225):
            if time.time() > deadline:
                break
            s = list(range(a, a + 32))
            if s[-1] > 256:
                break
            if fast_check(s):
                best_size = 32
                best_set = s
                break
        
        # Try APs with d=1,2,3
        for d in range(1, 4):
            if time.time() > deadline:
                break
            max_a = 256 - 31 * d
            for a in range(1, max_a + 1):
                s = [a + i * d for i in range(32)]
                if fast_check(s):
                    best_size = 32
                    best_set = s
                    break
            if best_size >= 32:
                break
    
    # Strategy 5: Use the bitset DP for more thorough validation of near-misses
    if best_size < 35 and time.time() < deadline - 20:
        # Try to find sets where the prefix-sum check is close but not quite valid
        # and use bitset to verify
        for target in range(32, min(best_size + 3, 40)):
            if time.time() > deadline:
                break
            # Try random sets
            for _ in range(500):
                if time.time() > deadline:
                    break
                # Generate a random set
                s = random.sample(range(1, 257), target)
                s.sort()
                if check_with_bitset(s):
                    if target > best_size:
                        best_size = target
                        best_set = s
                    break
    
    # Final validation
    if not fast_check(best_set):
        # Double-check with bitset
        if not check_with_bitset(best_set):
            return fallback
    
    # Ensure return type is correct
    result = [int(x) for x in best_set]
    return result
