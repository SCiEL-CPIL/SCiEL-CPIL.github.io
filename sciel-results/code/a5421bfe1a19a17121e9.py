import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe within 60s
    
    S = []
    
    # Pre-calculate max possible sum for bitset sizing?
    # Max element 512, max 47 elements. Sum < 512*47 = 24064.
    # We can use Python integers as bitsets.
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
            
        valid = True
        # Check against all existing y in S
        for y in S:
            # Construct S_prime = S \ {y}
            S_prime = [v for v in S if v != y]
            
            # DP for subset sums with cardinality
            # dp[k] is a bitset of achievable sums with k elements
            max_k = len(S_prime)
            dp = [0] * (max_k + 1)
            dp[0] = 1  # Sum 0 with 0 elements
            
            # Fill DP
            # Optimization: track max sum reached? Not strictly needed for Python big ints.
            for val in S_prime:
                # Iterate k backwards to avoid using same element multiple times in one step
                # We are iterating through elements one by one.
                # Standard 0/1 knapsack style.
                # For each element val, update dp[k] using dp[k-1] from PREVIOUS element.
                # So we need to iterate k from max_k down to 1.
                # But we are adding one element at a time.
                # So for a specific `val`, we update dp based on the state BEFORE adding `val`.
                # This requires iterating k in reverse.
                
                # Wait, the standard way is:
                # for val in items:
                #   for k in range(max_k, 0, -1):
                #      dp[k] |= dp[k-1] << val
                # This works because dp[k-1] on the RHS is the state from the previous iteration of the outer loop (i.e., without `val`).
                
                for k in range(max_k, 0, -1):
                    dp[k] |= (dp[k-1] << val)
            
            # Now check if any k >= 1 has a sum V = y(k+1) - x
            for k in range(1, max_k + 1):
                V = y * (k + 1) - x
                if V < 0:
                    continue
                # Check if bit V is set in dp[k]
                if (dp[k] >> V) & 1:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            S.append(x)
            if len(S) >= 48:
                break
                
    return S
