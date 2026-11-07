import random

def construct():
    max_val = 256
    max_size = 60
    
    def is_valid(S):
        # Check if S satisfies the condition
        # S_k disjoint for all k
        # Use bitsets for efficiency
        n = len(S)
        if n == 0:
            return True
        
        # dp[k] is a set of sums (using Python set or bitset)
        # Bitset is faster
        # Max sum is 60 * 256 = 15360
        # We can use integers as bitsets
        
        # However, we only need to check the final set.
        # We can recompute dp from scratch.
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for x in S:
            new_dp = dp[:]
            for k in range(n, 0, -1):
                new_dp[k] |= (dp[k-1] << x)
            dp = new_dp
            
        # Check disjointness
        for i in range(n + 1):
            if dp[i] == 0:
                continue
            for j in range(i + 1, n + 1):
                if dp[j] == 0:
                    continue
                if (dp[i] & dp[j]) != 0:
                    return False
        return True

    def try_greedy(candidates):
        S = []
        for x in candidates:
            if len(S) >= max_size:
                break
            S.append(x)
            if not is_valid(S):
                S.pop()
        return S

    # Strategy 1: Odd numbers
    odds = [i for i in range(1, max_val + 1, 2)]
    S1 = try_greedy(odds)
    
    # Strategy 2: Powers of 2 (known to work)
    pows = [2**i for i in range(0, 9)] # up to 256
    S2 = try_greedy(pows)
    
    # Strategy 3: Random search with restarts
    best_S = S1 if len(S1) > len(S2) else S2
    
    for _ in range(100):
        # Generate a random candidate set
        # We want distinct integers in 1..256
        # Try to construct a random set and validate?
        # Better: Greedy with random order
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        S_rand = try_greedy(candidates)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            if len(best_S) == max_size:
                break
                
    return best_S
