import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    max_n = 512
    max_size = 48
    
    # Precompute for each x, the number of 3-term APs where x is the middle
    # x is middle of (x-d, x, x+d) for d=1 to min(x-1, max_n-x)
    ap3_middle_counts = [0] * (max_n + 1)
    for x in range(1, max_n + 1):
        d_max = min(x - 1, max_n - x)
        ap3_middle_counts[x] = d_max
    
    # Precompute for each x, the number of pairs (a,b) with a+b = 2x
    # This is the same as ap3_middle_counts since a=x-d, b=x+d
    # So ap3_middle_counts[x] = min(x-1, max_n-x)
    
    # Strategy: Greedy with smarter ordering and better conflict checking
    # Key insight: elements with fewer potential conflicts should be added first
    # Elements near the edges have fewer 3-term AP conflicts
    
    def can_add(x, S, dp):
        """
        Checks if x can be added to S without violating constraints.
        dp[k] is a bitset of achievable sums using k elements from S.
        Returns True if x can be added, False otherwise.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # sum(T) = x * |T|
        for k in range(2, n_s + 1):
            target_sum = x * k
            # Max possible sum of k elements from S is sum of k largest elements
            # But we can just check if target_sum is achievable
            if target_sum > 512 * n_s:  # Max possible sum
                break
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y is mean of T' = T U {x}, T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        # We check if sum(T) is achievable by j elements from S.
        # Conservative: If achievable from S, assume it's achievable from S\{y} (reject).
        for y in S:
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < 0:
                    continue
                if target_sum > 512 * n_s:
                    break
                if (dp[j] >> target_sum) & 1:
                    return False
                    
        return True
    
    def greedy_build(order):
        """
        Builds a set using the given order of candidates.
        Returns the set of integers.
        """
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
                
            if can_add(x, S, dp):
                S.append(x)
                # Update DP
                # Iterate backwards to avoid using x multiple times
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return S
    
    def try_fill(S):
        """
        Attempts to add more elements to S by scanning all candidates.
        """
        if time.time() > deadline - 5.0:
            return S
            
        S = list(S)
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
                
        changed = True
        while changed and len(S) < max_size:
            changed = False
            for x in range(1, max_n + 1):
                if x in S:
                    continue
                if time.time() > deadline - 5.0:
                    break
                if can_add(x, S, dp):
                    S.append(x)
                    for k in range(len(S) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    changed = True
                    break # Restart scan to ensure consistency
        return S
    
    best_set = []
    best_size = 0
    
    # Heuristic 1: Edge-First (elements near edges have fewer 3-AP conflicts)
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    S_edge = greedy_build(order_edge)
    best_set = S_edge
    best_size = len(best_set)
    
    # Heuristic 2: Safe-First (Min 3-AP middle count)
    order_safe = sorted(range(1, max_n + 1), key=lambda x: ap3_middle_counts[x])
    S_safe = greedy_build(order_safe)
    if len(S_safe) > best_size:
        best_set = S_safe
        best_size = len(best_set)
    
    # Heuristic 3: Combined - prioritize elements with low conflict potential
    # Score: lower is better (fewer conflicts)
    def combined_score(x):
        # Weight 3-AP conflicts heavily, add small penalty for being central
        edge_dist = abs(x - 256.5)
        return ap3_middle_counts[x] * 10 - edge_dist
        
    order_combined = sorted(range(1, max_n + 1), key=combined_score)
    S_comb = greedy_build(order_combined)
    if len(S_comb) > best_size:
        best_set = S_comb
        best_size = len(best_set)
    
    # Heuristic 4: Start with extremes, then fill in
    # Take smallest and largest alternately
    order_extremes = []
    left, right = 1, max_n
    while left <= right:
        order_extremes.append(left)
        if left != right:
            order_extremes.append(right)
        left += 1
        right -= 1
    S_extremes = greedy_build(order_extremes)
    if len(S_extremes) > best_size:
        best_set = S_extremes
        best_size = len(best_set)
    
    # Randomized Search with Heuristic Bias
    random.seed(42)
    
    safe_list = order_safe[:]
    
    # Try multiple strategies:
    # 1. Pure random
    # 2. Random with safe bias
    # 3. Random with edge bias
    
    strategies = [
        ("random", lambda: list(random.sample(range(1, max_n + 1), max_n))),
        ("safe_bias", lambda: safe_list[:100] + random.sample(range(1, max_n + 1), max_n - 100)),
        ("edge_bias", lambda: order_edge[:100] + random.sample(range(1, max_n + 1), max_n - 100)),
    ]
    
    attempt_count = 0
    while time.time() < deadline - 10.0:
        attempt_count += 1
        strategy_name = strategies[attempt_count % len(strategies)][0]
        order = strategies[attempt_count % len(strategies)][1]()
        
        S_rand = greedy_build(order)
        
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break
    
    # Final Fill
    if time.time() < deadline - 5.0:
        filled_set = try_fill(best_set)
        if len(filled_set) > best_size:
            best_set = filled_set
            best_size = len(filled_set)
    
    # If still not at max, try more aggressive random restarts
    if len(best_set) < max_size and time.time() < deadline - 2.0:
        while time.time() < deadline - 1.0:
            order = list(range(1, max_n + 1))
            random.shuffle(order)
            S_rand = greedy_build(order)
            if len(S_rand) > best_size:
                best_set = S_rand
                best_size = len(best_set)
                if best_size >= max_size:
                    break
    
    return best_set
