import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe within 60s

    # Generate candidates: Numbers with base-3 digits only 0 or 1
    # This set is 3-AP-free, which is a necessary condition.
    candidates = []
    for i in range(1, 513):
        n = i
        valid_base3 = True
        while n > 0:
            if n % 3 == 2:
                valid_base3 = False
                break
            n //= 3
        if valid_base3:
            candidates.append(i)
    
    candidates.sort()
    
    # Helper to check if a set S is valid
    def is_valid_set(S):
        if not S:
            return True
        n = len(S)
        # For each x in S, check if x is mean of subset of S\{x}
        # DP: dp[k] is bitset of sums achievable with k elements
        # Max sum < 512 * 48 = 24576
        # We check for each x
        for x in S:
            S_prime = [v for v in S if v != x]
            if not S_prime:
                continue
            
            max_k = len(S_prime)
            dp = [0] * (max_k + 1)
            dp[0] = 1  # Sum 0 with 0 elements
            
            for val in S_prime:
                for k in range(max_k, 0, -1):
                    dp[k] |= (dp[k-1] << val)
            
            # Check if any k >= 2 has sum k*x
            for k in range(2, max_k + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    return False
        return True

    # Try the full base-3 set (or first 48)
    # The base-3 set has ~63 elements.
    # We want size 48.
    # If the first 48 are valid, we are done.
    # But maybe a different subset of 48 is valid?
    # The base-3 set is structured. Taking the smallest 48 is a good bet.
    
    S = candidates[:48]
    
    if is_valid_set(S):
        return S

    # If the simple prefix fails, try a greedy selection from candidates
    # Prioritize smaller numbers? Or random?
    # Since candidates are 3-AP-free, only k>=3 is the issue.
    # Let's try to build a set greedily from candidates.
    
    S_greedy = []
    for x in candidates:
        if len(S_greedy) >= 48:
            break
        if time.time() > deadline:
            break
        
        # Check if x can be added to S_greedy
        # We need to check if x is mean of subset of S_greedy (k>=2)
        # AND if any element in S_greedy is mean of subset of S_greedy + {x} (k>=2)
        
        # Actually, just check the full set S_greedy + [x]
        # But that's expensive.
        # Optimization:
        # 1. Check if x is mean of subset of S_greedy.
        # 2. Check if any y in S_greedy is mean of subset of (S_greedy \ {y}) + {x}.
        
        # Let's just use the full check for simplicity, as N is small (<=48)
        # But full check is O(N * 2^N) roughly? No, DP is O(N * Sum).
        # Sum ~ 12000. N=48. 48*12000 = 576k ops. Very fast.
        
        new_S = S_greedy + [x]
        if is_valid_set(new_S):
            S_greedy.append(x)
            
    if len(S_greedy) >= 48:
        return S_greedy[:48]

    # Fallback: Global Greedy with Conflict Heuristic
    # If base-3 didn't work, try all numbers 1..512
    # Sort by "conflict count" (number of 3-APs they form in [1,512])
    
    # Precompute conflict counts (3-AP middle count)
    # x is middle if 2x = a+b.
    # Count pairs (a,b) in [1,512] such that 2x = a+b.
    # This is just the number of solutions.
    # For small x, few solutions. For x=256, many.
    # So sort by x ascending?
    # The base-3 set is already sorted ascending.
    
    # Let's try a random shuffle of 1..512 and greedy add.
    import random
    random.seed(42)
    all_nums = list(range(1, 513))
    random.shuffle(all_nums)
    
    S_random = []
    for x in all_nums:
        if len(S_random) >= 48:
            break
        if time.time() > deadline:
            break
            
        new_S = S_random + [x]
        if is_valid_set(new_S):
            S_random.append(x)
            
    if len(S_random) >= 48:
        return S_random[:48]
        
    # Return best found
    if len(S_greedy) > len(S_random):
        return S_greedy
    return S_random
