"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    # We want to find the minimum number of distinct divisors that sum to target.
    # dp[s] = (min_count, bitmask) where bitmask indicates which divisors are used.
    # However, storing bitmasks for all sums up to 4979 with 60 divisors might be memory heavy if we store full bitmasks.
    # Instead, we can use a dictionary or array where dp[s] stores the minimum count, and we track the parent to reconstruct,
    # OR we can store the bitmask directly since 60 bits fits in an integer.
    
    # Let's use an array for dp where index is sum.
    # dp[s] = bitmask of divisors used, or -1 if unreachable.
    # We also need to track the count to minimize it.
    # Since we want minimum cardinality, we can iterate by number of items?
    # Or we can do standard DP where we update dp[s + d] from dp[s] if count is lower.
    
    # Let's use: dp[s] = (count, bitmask)
    # Initialize dp[0] = (0, 0)
    # For each divisor d, we can choose to include it or not.
    # But standard 0/1 knapsack style:
    # Iterate through divisors. For each divisor, update dp in reverse order of sums to avoid reusing the same divisor.
    
    # Max sum is target = 4979.
    # dp[s] will store the bitmask. We can infer count from the bitmask if we just want to check, but we need to minimize count.
    # Actually, if we just store the bitmask, we can compare popcount.
    # Let's store dp[s] as an integer bitmask. If unreachable, store -1.
    # When updating, if new_count < old_count, update.
    
    import math
    
    # Initialize dp array with -1 (unreachable)
    dp = [-1] * (target + 1)
    dp[0] = 0  # bitmask 0 means no divisors chosen
    
    # Iterate through each divisor
    # To ensure we find the global minimum, we process divisors one by one.
    # For each divisor d_i, we update dp[s + d_i] from dp[s] for all s from target down to 0.
    # If dp[s] is not -1, then we can form s + d_i using the divisors in dp[s] plus d_i.
    # We compare the popcount of (dp[s] | (1 << i)) with the current popcount of dp[s + d_i].
    # If it's smaller, we update.
    
    # Note: We must be careful with the order. By iterating s from high to low, we ensure each divisor is used at most once.
    
    for i, d in enumerate(divisors):
        bit = 1 << i
        # Iterate s from target down to 0
        # If s + d > target, we can skip, so max s is target - d
        for s in range(target - d, -1, -1):
            if dp[s] != -1:
                new_mask = dp[s] | bit
                # Calculate popcount of new_mask
                new_count = bin(new_mask).count('1')
                
                # Check current state of dp[s+d]
                if dp[s + d] == -1:
                    dp[s + d] = new_mask
                else:
                    current_count = bin(dp[s + d]).count('1')
                    if new_count < current_count:
                        dp[s + d] = new_mask
    
    if dp[target] == -1:
        # Should not happen for practical numbers, but just in case
        raise ValueError("No solution found")
    
    # Reconstruct the divisors from the bitmask
    result_mask = dp[target]
    selected_divisors = []
    for i in range(num_divs):
        if result_mask & (1 << i):
            selected_divisors.append(divisors[i])
    
    # Verify sum and distinctness
    assert len(selected_divisors) == len(set(selected_divisors))
    assert sum(selected_divisors) == target
    assert all(n % d == 0 for d in selected_divisors)
    
    return selected_divisors
# EVOLVE-BLOCK-END
