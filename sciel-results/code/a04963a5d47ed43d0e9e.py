import time
import random

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0

    max_n = 512
    max_size = 48
    max_sum = max_n * max_size  # 24576

    # Precompute static safety scores:
    # Count of pairs (u, v) with u + v = 2x, u != v, u,v != x
    # This approximates the number of 3-term APs centered at x
    safety_scores = [0] * (max_n + 1)
    for x in range(1, max_n + 1):
        count = 0
        # u < x < v, u + v = 2x
        # u ranges from max(1, 2x - max_n) to x - 1
        start_u = max(1, 2 * x - max_n)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        safety_scores[x] = count

    # Orders for greedy construction
    order_safe = sorted(range(1, max_n + 1), key=lambda x: safety_scores[x])
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    order_asc = list(range(1, max_n + 1))
    order_desc = list(range(max_n, 0, -1))

    def build_dp(S):
        """Build dp table for set S. dp[k] is bitset of sums."""
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            # Update in reverse to avoid using x multiple times
            # We only need up to max_size, but S might be smaller
            # We iterate up to current len(S)
            for k in range(len(S) - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << x)
        return dp

    def check_valid(x, S, dp):
        """
        Check if x can be added to S.
        S is a list, dp is the DP table for S.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # sum(T) = x * |T|
        for k in range(2, n_s + 1):
            target_sum = x * k
            if target_sum > max_sum:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y = mean(T' = T U {x}), T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        for y in S:
            # Optimization: if y is very small or large, some j values are impossible
            # target_sum = y * (j + 1) - x
            # Min sum for j elements is j * 1 = j
            # Max sum for j elements is j * 512
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < j or target_sum > j * max_n:
                    continue
                if target_sum > max_sum:
                    break # Assuming y is positive, larger j means larger target
                if (dp[j] >> target_sum) & 1:
                    return False
                    
        return True

    def greedy_build(order, S_init=None):
        """
        Builds a set using the given order of candidates.
        """
        S = list(S_init) if S_init else []
        S_set = set(S)
        
        if not S:
            dp = [0] * (max_size + 1)
            dp[0] = 1
        else:
            dp = build_dp(S)
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
            if x in S_set:
                continue
            
            if check_valid(x, S, dp):
                S.append(x)
                S_set.add(x)
                # Update DP incrementally
                for k in range(len(S) - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
                        
        return S

    def local_search(S):
        """
        Attempts to improve the set S by adding elements or swapping.
        """
        if time.time() > deadline - 10.0:
            return S
            
        S = list(S)
        S_set = set(S)
        dp = build_dp(S)
        
        improved = True
        while improved and len(S) < max_size:
            improved = False
            # Try to add any valid element
            # We scan all candidates
            for x in range(1, max_n + 1):
                if x in S_set:
                    continue
                if time.time() > deadline - 5.0:
                    return S
                if check_valid(x, S, dp):
                    S.append(x)
                    S_set.add(x)
                    for k in range(len(S) - 1, -1, -1):
                        if dp[k]:
                            dp[k + 1] |= (dp[k] << x)
                    improved = True
                    break # Restart scan as DP changed
        
        if len(S) == max_size:
            return S
            
        # If stuck, try swaps
        # We perform a number of random swap attempts
        max_swaps = 1000
        for _ in range(max_swaps):
            if time.time() > deadline - 5.0:
                return S
                
            # Pick a random element from S to remove
            idx = random.randint(0, len(S) - 1)
            removed = S[idx]
            
            # Remove it
            S.pop(idx)
            S_set.remove(removed)
            dp = build_dp(S) # Rebuild DP
            
            # Try to add any valid element
            added = -1
            # We can prioritize candidates that are "safe" or "edge"
            # But random is fine for diversity
            candidates = [x for x in range(1, max_n + 1) if x not in S_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if check_valid(x, S, dp):
                    added = x
                    break
            
            if added != -1:
                # Accept swap
                S.append(added)
                S_set.add(added)
                # Update DP
                for k in range(len(S) - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << added)
                
                # After a successful swap, we might be able to add more
                # So we break out of the swap loop and go back to the "add" loop
                # But the "add" loop is inside the while loop above.
                # So we set improved = True and break the swap loop?
                # No, the structure is:
                # while improved:
                #   try add
                #   if not full, try swaps
                # 
                # If we accept a swap, we should restart the "try add" phase.
                # So we break the swap loop and let the while loop continue.
                improved = True
                break
            else:
                # Revert
                S.append(removed)
                S_set.add(removed)
                dp = build_dp(S)
                
        return S

    best_set = []
    best_size = 0

    # 1. Deterministic Greedies
    S1 = greedy_build(order_safe)
    best_set = S1
    best_size = len(best_set)
    
    S2 = greedy_build(order_edge)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(best_set)
        
    S3 = greedy_build(order_desc)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(best_set)

    S4 = greedy_build(order_asc)
    if len(S4) > best_size:
        best_set = S4
        best_size = len(best_set)

    # 2. Local Search on Best Set
    if time.time() < deadline - 15.0:
        S_ls = local_search(best_set)
        if len(S_ls) > best_size:
            best_set = S_ls
            best_size = len(best_set)

    # 3. Randomized Restarts
    random.seed(42)
    while time.time() < deadline - 10.0:
        # Create a noisy safe order
        order = list(order_safe)
        # Shuffle the bottom 50% to introduce variation
        half = max_n // 2
        random.shuffle(order[half:])
        
        S_rand = greedy_build(order)
        
        # Apply local search if it's promising (close to best)
        if len(S_rand) >= best_size - 2 and len(S_rand) < max_size:
            if time.time() < deadline - 5.0:
                S_rand = local_search(S_rand)
                
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break

    return best_set
