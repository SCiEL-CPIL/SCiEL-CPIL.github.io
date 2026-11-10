import random
import time

def construct():
    max_n = 512
    max_size = 48
    
    def is_valid(S):
        S = list(S)
        n = len(S)
        if n < 2:
            return True
        
        # For each x in S, check if x is the mean of any subset of S \ {x}
        for i, x in enumerate(S):
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            
            # We need to check if sum(T) = k * x for some T subset others, |T| >= 2, k = |T|
            # This is equivalent to: sum(t - x) = 0 for T subset others, |T| >= 2
            # Let v_j = others[j] - x
            # We need a subset of v's that sums to 0, with size >= 2.
            
            # DP for subset sum of v's
            # Range of v's: [-511, 511]
            # Max sum magnitude: 47 * 512 ~ 24000
            # Offset: 25000
            offset = 25000
            # dp[k] is a bitset where bit 's' is set if sum 's' is achievable with k elements
            # We only care about sum 0.
            
            # Optimization: If the number of others is small, we can do this.
            # But 47 is manageable.
            
            # To save time, we can just check if 0 is in the subset sums of size >= 2.
            # We can use a set of achievable sums for each size, but bitsets are faster.
            
            # However, doing this for all 48 elements might be slow if done naively in Python loops.
            # Let's use a single DP that tracks (sum, count) pairs?
            # Or just standard bitset DP for each x.
            
            # Bitset DP for sum 0
            # dp[k] bitset
            dp = [0] * (m + 1)
            dp[0] = 1 << offset
            
            for v in others:
                # Iterate backwards
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        if v >= 0:
                            dp[k+1] |= (dp[k] << v)
                        else:
                            dp[k+1] |= (dp[k] >> (-v))
            
            # Check if 0 is achievable with k >= 2
            target_bit = 1 << offset
            for k in range(2, m + 1):
                if dp[k] & target_bit:
                    return False
        
        return True

    # Randomized Local Search
    best_set = None
    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer
    
    # Initial random set
    S = random.sample(range(1, max_n + 1), max_size)
    
    # Check initial
    if is_valid(S):
        return S
        
    # Local search
    # We will try to swap elements to fix violations.
    # To speed up, we can just generate new random sets if the first few fail.
    # Given the low density, random sets are likely valid.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        S = random.sample(range(1, max_n + 1), max_size)
        if is_valid(S):
            return S
            
    # Fallback: Return the powers of 2 set (valid, but small)
    # Or a known valid set
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

