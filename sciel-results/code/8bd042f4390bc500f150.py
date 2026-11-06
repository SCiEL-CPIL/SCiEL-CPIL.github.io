import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_n = 512
    max_size = 48
    
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
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y is mean of T' = T U {x}, T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        # We check if sum(T) is achievable by j elements from S.
        # Conservative: If achievable from S, assume it's achievable from S\{y} (reject).
        # This ensures validity.
        for y in S:
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < 0:
                    continue
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
    
    # 1. Deterministic Greedy
    # Descending often works well to avoid small numbers being means of many
    order_desc = range(max_n, 0, -1)
    S_desc = greedy_build(order_desc)
    best_set = S_desc
    best_size = len(best_set)
    
    # Ascending
    order_asc = range(1, max_n + 1)
    S_asc = greedy_build(order_asc)
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        
    # 2. Randomized Greedy
    random.seed(42)
    
    # Run multiple random shuffles
    # We can run more if we are fast
    while time.time() < deadline - 15.0:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S_rand = greedy_build(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break
                
    # 3. Try to fill gaps in the best set found
    if time.time() < deadline - 5.0:
        filled_set = try_fill(best_set)
        if len(filled_set) > best_size:
            best_set = filled_set
            best_size = len(best_set)
            
    # 4. Final check and return
    # The greedy and fill methods are conservative, so best_set should be valid.
    # We return the list of integers.
    return best_set
