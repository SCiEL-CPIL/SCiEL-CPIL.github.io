import time
import random

def _is_valid_set(S):
    """
    Checks if no element in S is the mean of a subset of at least 2 other elements.
    Uses bitset DP for subset sums with exact count.
    """
    n = len(S)
    if n < 3:
        return True
        
    for i, a in enumerate(S):
        # Others are all elements except a
        others = [x for j, x in enumerate(S) if j != i]
        
        # Max possible sum of 'k' elements from others
        # We need to check if k*a is a subset sum of 'others' using exactly k elements.
        # k can range from 2 to len(others)
        
        # DP: dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        # Max sum is sum(others). Let's estimate max sum to limit bitset size if needed, 
        # but Python big ints handle this dynamically.
        
        # Initialize dp
        # dp[0] = 1 (bit 0 set, sum 0 with 0 elements)
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            # Update from high count to low count to avoid using same element twice
            # Max count we care about is len(others)
            for j in range(len(others) - 1, -1, -1):
                if dp[j]:
                    dp[j+1] |= (dp[j] << x)
                    
        # Now check for any k >= 2 if k*a is in dp[k]
        # Note: a is not in others, so we are checking subsets of others.
        # We need to ensure the subset has at least 2 elements.
        for k in range(2, len(others) + 1):
            target_sum = k * a
            # Check if bit target_sum is set in dp[k]
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    best_set = []
    
    # Strategy: Try multiple greedy passes with different ordering heuristics.
    # 1. Increasing order
    # 2. Decreasing order
    # 3. Randomized orders
    
    orders = []
    
    # 1. Increasing
    orders.append(list(range(1, 513)))
    
    # 2. Decreasing
    orders.append(list(range(512, 0, -1)))
    
    # 3. Randomized (a few seeds)
    rng = random.Random(42)
    for _ in range(5):
        cand = list(range(1, 513))
        rng.shuffle(cand)
        orders.append(cand)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = []
        for c in order:
            if time.time() > deadline:
                break
                
            if c in S:
                continue
                
            new_S = S + [c]
            if _is_valid_set(new_S):
                S = new_S
            if len(S) == 48:
                break
                
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
