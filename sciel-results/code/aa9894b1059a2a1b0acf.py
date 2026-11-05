import time
import random

def is_valid(S):
    n = len(S)
    if n < 2:
        return True
    S_set = set(S)
    for x in S:
        others = [y for y in S if y != x]
        max_sum = sum(others)
        # dp[k] is a bitset of sums achievable with k elements from 'others'
        # We only need to track sums up to max_sum
        dp = [0] * (len(others) + 1)
        dp[0] = 1  # sum 0 is possible with 0 elements
        
        for y in others:
            # Iterate backwards to ensure each element is used at most once
            for k in range(len(others) - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << y)
        
        # Check if x is the mean of any subset of 'others' of size k >= 2
        for k in range(2, len(others) + 1):
            target_sum = k * x
            if target_sum > max_sum:
                break
            # Check if target_sum is achievable with k elements
            if (dp[k] >> target_sum) & 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Attempt 1: Random sampling
    # A random set of 48 distinct integers from [1, 512]
    # Probability of being valid is unknown but likely non-zero.
    for _ in range(500):
        if time.time() > deadline:
            break
        S = random.sample(range(1, 513), 48)
        if is_valid(S):
            return S

    # Attempt 2: Greedy construction
    # Try to build a set of size 48 by adding elements one by one.
    best_S = []
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        S = []
        available = list(range(1, 513))
        random.shuffle(available)
        
        for x in available:
            if len(S) == 48:
                break
            if time.time() > deadline:
                break
            
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        
        if len(S) > len(best_S):
            best_S = S
        
        if len(S) == 48:
            return S

    # Fallback: Return the best set found, or a small known valid set
    if not best_S:
        # Powers of 2 up to 256 plus 512 is valid (size 10)
        # Let's just return a simple valid set to ensure non-zero score if all else fails
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    return best_S
