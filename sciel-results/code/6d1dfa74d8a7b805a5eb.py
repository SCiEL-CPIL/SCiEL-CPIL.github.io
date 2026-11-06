import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    def is_valid_addition(existing, candidate, max_val=512):
        """
        Check if adding 'candidate' to 'existing' is valid.
        For each element 'a' in existing + [candidate], check if 'a' is the mean
        of any subset of at least two OTHER elements.
        """
        # We need to check two cases:
        # 1. 'candidate' is the mean of a subset of 'existing' of size >= 2
        # 2. Some 'a' in 'existing' is the mean of a subset of 'existing' + [candidate]
        #    (excluding 'a' itself) of size >= 2
    
    # Let's use a more efficient approach:
    # For a set S, an element a is invalid if there exists a subset T of S\{a}, |T|>=2, such that sum(T) = a*|T|
    # This is equivalent to: sum(T \ a) = 0, i.e., the signed sum of elements in T with a as the mean is 0.
    # More precisely: sum(t for t in T) = a * |T| => sum(t - a for t in T) = 0
    
    # We can use subset sum DP on the transformed values (t - a).
    # But since we're adding one element at a time, we need to be careful.
    
    # Actually, let's think differently. When we add a new element x to a set S:
    # - We need to verify that x is NOT the mean of any subset of S with size >= 2
    # - We need to verify that no element a in S becomes the mean of a subset of S ∪ {x} \ {a} with size >= 2
    #   This could happen if x is part of such a subset.
    
    # For the first check: x is mean of subset T of S, |T|>=2, sum(T) = x*|T|
    #   => sum(t - x for t in T) = 0
    #   We can do subset sum on (t - x) for t in S.
    
    # For the second check: a in S is mean of subset U of (S ∪ {x}) \ {a}, |U|>=2, sum(U) = a*|U|
    #   This was already checked when we built S (for subsets not containing x).
    #   New subsets containing x: U = V ∪ {x} where V ⊆ S \ {a}, and sum(V) + x = a*(|V|+1)
    #   => sum(v - a for v in V) = a - x
    #   So we need to check if there exists V ⊆ S\{a}, |V| >= 1 (so |U| = |V|+1 >= 2), such that sum(v-a for v in V) = a-x
    
    # This is getting complex. Let's use a simpler but still efficient approach.
    # Given the small size (max 48 elements, values up to 512), we can use bitset DP.
    
    def check_no_mean(existing, candidate):
        """Check if candidate is not the mean of any subset of existing with size >= 2."""
        if len(existing) < 2:
            return True
        # We need sum(T) = candidate * |T| for some T ⊆ existing, |T| >= 2
        # => sum(t - candidate for t in T) = 0
        # Use DP on values (t - candidate)
        # Range of (t - candidate): from 1-512=-511 to 512-1=511
        # Sum range: for |T| up to len(existing), min sum ~ -511*48, max ~ 511*48
        # That's too large. Let's use a different approach.
        
        # Alternative: For each possible subset size k >= 2, check if there's a subset of size k with sum = candidate * k
        # We can use DP: dp[k][s] = True if there's a subset of size k with sum s
        # But s can be up to 512*48 = 24576, and k up to 48.
        # That's 48 * 24576 ~ 1.2M states, which is feasible.
        
        max_k = len(existing)
        max_sum = 512 * max_k
        
        # dp[k] will be a bitset (or set) of achievable sums for subset size k
        # Use a list of sets for simplicity, or bitarrays
        # Let's use a list of boolean arrays or sets
        
        # Actually, let's use a dictionary of sets: dp[k] = set of achievable sums for subset size k
        # But this might be slow. Let's use bitarrays with Python integers (bitsets)
        
        # dp[k] is an integer where bit s is set if sum s is achievable with subset size k
        # s ranges from 0 to max_sum
        # max_sum = 512 * 48 = 24576, so we need 24577 bits ~ 3072 bytes, which is fine.
        
        dp = [0] * (max_k + 1)
        dp[0] = 1  # sum 0 is achievable with 0 elements
        
        for x in existing:
            # Update dp in reverse order of k
            for k in range(len(existing) - 1, -1, -1):
                if k < max_k:
                    # dp[k+1] |= dp[k] << x
                    dp[k+1] |= (dp[k] << x)
        
        # Check for k >= 2: is bit (candidate * k) set in dp[k]?
        for k in range(2, max_k + 1):
            target_sum = candidate * k
            if 0 <= target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
        return True
    
    def check_existing_still_valid(existing, candidate):
        """Check that no element in existing becomes the mean of a subset involving candidate."""
        # For each a in existing, check if there's a subset U of (existing ∪ {candidate}) \ {a} with |U| >= 2 and sum(U) = a * |U|
        # New subsets must contain candidate. So U = V ∪ {candidate} where V ⊆ existing \ {a}, |V| >= 1, and sum(V) + candidate = a * (|V| + 1)
        # => sum(v - a for v in V) = a - candidate
        for a in existing:
            others = [x for x in existing if x != a]
            # We need V ⊆ others, |V| >= 1, sum(v - a for v in V) = a - candidate
            # Let w = v - a for v in others. We need a subset of w with sum = a - candidate and size >= 1.
            if not others:
                continue
            target = a - candidate
            # Use DP on w values
            # w ranges from 1-a to 512-a
            # Sum range is bounded
            
            # Use bitset DP: dp[s] = bitset of achievable sizes for sum s
            # Or: dp[k] = bitset of achievable sums for size k
            max_k = len(others)
            # Compute min and max possible sums
            w_vals = [v - a for v in others]
            min_sum = sum(w for w in w_vals if w < 0)
            max_sum = sum(w for w in w_vals if w > 0)
            
            if target < min_sum or target > max_sum:
                continue
            
            offset = -min_sum if min_sum < 0 else 0
            range_size = max_sum - min_sum + 1
            
            if range_size > 100000:  # Safety check
                # Fallback to slower method or skip
                continue
            
            # dp[k] is a bitset (Python int) where bit (s - min_sum) is set if sum s is achievable with size k
            dp = [0] * (max_k + 1)
            dp[0] = 1  # sum 0 (offset 0) achievable with size 0
            
            for w in w_vals:
                for k in range(len(others) - 1, -1, -1):
                    if k < max_k:
                        dp[k+1] |= (dp[k] << w)
            
            # Check if any k >= 1 has bit (target - min_sum) set
            idx = target - min_sum
            if 0 <= idx < range_size:
                for k in range(1, max_k + 1):
                    if (dp[k] >> idx) & 1:
                        return False
        return True
    
    def greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if check_no_mean(selected, x) and check_existing_still_valid(selected, x):
                selected.append(x)
        return selected
    
    best = []
    
    # Try multiple orders
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(5):
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        result = greedy(order)
        if len(result) > len(best):
            best = result
        if len(best) >= 48:
            break
    
    return best
