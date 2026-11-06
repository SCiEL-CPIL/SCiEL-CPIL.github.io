import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    max_n = 512
    max_size = 48

    # Precompute 3-AP counts
    ap3_counts = [0] * (max_n + 1)
    for x in range(1, max_n + 1):
        count = 0
        start_u = max(1, 2*x - max_n)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        ap3_counts[x] = count
        
    # Precompute 4-AP counts (x is mean of 4 others? No, x is mean of T, |T|=4)
    # sum(T) = 4x.
    # This is harder to count simply. Let's stick to 3-APs and edge distance.
    
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
            if target_sum > 512 * n_s: # Max possible sum
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

    # Heuristic 1: Edge-First
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    S_edge = greedy_build(order_edge)
    best_set = S_edge
    best_size = len(best_set)

    # Heuristic 2: Safe-First (Min 3-AP)
    order_safe = sorted(range(1, max_n + 1), key=lambda x: ap3_counts[x])
    S_safe = greedy_build(order_safe)
    if len(S_safe) > best_size:
        best_set = S_safe
        best_size = len(S_safe)

    # Heuristic 3: Combined Score
    # Score = ap3_count + 0.1 * (100 - edge_dist)
    def combined_score(x):
        edge_dist = abs(x - 256.5)
        return ap3_counts[x] * 10 - edge_dist
        
    order_combined = sorted(range(1, max_n + 1), key=combined_score)
    S_comb = greedy_build(order_combined)
    if len(S_comb) > best_size:
        best_set = S_comb
        best_size = len(S_comb)

    # Randomized Search with Heuristic Bias
    random.seed(42)
    
    # Try multiple random restarts with noise added to the safe order
    # Or just random shuffles, but biased towards safe numbers.
    
    # Strategy: 
    # 1. Take the top K safest numbers.
    # 2. Shuffle the rest.
    # 3. Concatenate.
    
    safe_list = order_safe[:]
    
    while time.time() < deadline - 10.0:
        # Vary the bias
        bias_factor = random.uniform(0.5, 1.0)
        num_safe = int(max_n * bias_factor)
        
        top_safe = safe_list[:num_safe]
        rest = safe_list[num_safe:]
        random.shuffle(rest)
        
        # Add some random noise to top_safe as well?
        # Just shuffle a portion of top_safe
        if random.random() < 0.5:
            random.shuffle(top_safe)
            
        order = top_safe + rest
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

    # Local Search: Try to swap elements to improve "potential"
    # If we are stuck, try to swap an element with a "safer" one.
    # This is complex to implement correctly with DP updates.
    # Instead, we do more random restarts if time permits.
    
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
            # Try fill on random set too? Might be too slow.
            
    return best_set
