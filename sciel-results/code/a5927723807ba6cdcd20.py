"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    # Denominators are 2, 3, ..., 48
    # Zero-based indices 0..46 correspond to denominators 2..48
    n = 47
    target = Fraction(1, 1)
    
    # dp[k] will store the minimum sum (strictly < 1) achievable with exactly k items
    # We use None to represent infinity (unreachable)
    # We iterate through denominators one by one (0/1 knapsack style)
    # To save memory and time, we only need to keep track of the best (minimum) sum for each count k.
    # Since we want to minimize the deficit (1 - sum), we want to maximize the sum < 1.
    # Wait, the task says "Minimize positive deficit 1-sum".
    # Deficit = 1 - sum. Minimizing deficit is equivalent to maximizing sum, subject to sum < 1.
    
    # So we want the maximum sum < 1.
    # Let's adjust DP:
    # dp[k] = maximum sum < 1 achievable with exactly k items.
    # If no such sum exists, dp[k] = None.
    
    # Initialize dp
    # dp[0] = 0 (sum of 0 items is 0, which is < 1)
    dp = [None] * (n + 1)
    dp[0] = Fraction(0, 1)
    
    # We process each denominator
    for i in range(n):
        d = i + 2  # Denominator value
        val = Fraction(1, d)
        
        # Update dp in reverse order to avoid using the same item twice
        # We can only add val to dp[k-1] if dp[k-1] is not None and dp[k-1] + val < 1
        # And we want to maximize the resulting sum.
        
        # We iterate k from n down to 1
        # Actually, we only need to go up to i+1 since we've only processed i+1 items so far.
        # But n is small (47), so full range is fine.
        
        for k in range(i + 1, 0, -1):
            if dp[k - 1] is not None:
                new_sum = dp[k - 1] + val
                if new_sum < 1:
                    # Check if this new sum is better (larger) than current dp[k]
                    if dp[k] is None or new_sum > dp[k]:
                        dp[k] = new_sum
                    # Note: If new_sum >= 1, we discard it because the sum must be STRICTLY BELOW 1.
                    # If dp[k] is already a valid sum, we keep it.
                    # If dp[k] is None, we leave it None (unless new_sum is valid).
                    # If new_sum is valid but smaller than dp[k], we keep dp[k].
    
    # Now, find the k that gives the maximum sum < 1 (which minimizes the deficit)
    best_sum = None
    best_k = -1
    for k in range(n + 1):
        if dp[k] is not None:
            if best_sum is None or dp[k] > best_sum:
                best_sum = dp[k]
                best_k = k
    
    if best_k == -1:
        # Should not happen, but return empty list if no solution found
        return []
    
    # Now we need to reconstruct the set of indices that yields best_sum with best_k items.
    # We need to backtrack through the DP.
    # However, the standard DP state only stores the value, not the path.
    # We can re-run a similar DP but storing the parent pointer or the item used.
    # Given the small size (n=47), we can store the entire history or just recompute.
    # Let's store the item index used to reach the optimal state for each (i, k).
    # parent[i][k] = index of the item (0..i-1) that was added to reach state (i, k) from (i-1, k-1).
    # Or simply: did we include item i-1?
    
    # Let's do a second pass to track the choices.
    # dp_val[i][k] = max sum < 1 using first i items with exactly k of them.
    # parent[i][k] = True if item i-1 was included, False otherwise.
    
    # To save memory, we can just store the final best_k and best_sum, and then backtrack?
    # Backtracking is tricky without storing the full table.
    # Let's store the full table of booleans for inclusion.
    # n=47, so table is 48x48. Very small.
    
    # Re-initialize with tracking
    # dp_val[i][k]
    # dp_inc[i][k] : 1 if item i-1 is included in the optimal set for (i,k), 0 otherwise.
    # Note: If there are ties, we just pick one.
    
    # Initialize
    # i=0: 0 items processed.
    # dp_val[0][0] = 0
    # dp_val[0][k>0] = None
    
    # We'll use lists of lists.
    # dp_val[i] is a list of length n+1
    # dp_inc[i] is a list of length n+1
    
    # To optimize time, we can just re-run the DP but store the decision.
    
    # Let's just store the decision for the optimal path.
    # Actually, we can just store the set of indices for the best solution found in the first pass?
    # No, the first pass only found the value.
    
    # Let's do the second pass.
    # Time limit is 60s. 47*47*47 operations is negligible.
    
    # dp_val[i][k]
    # i from 0 to n
    # k from 0 to i
    
    # We only need to keep the previous row of dp_val to compute the current row, but we need the full history for backtracking.
    # So we store all rows.
    
    # dp_val[0] = [0, None, None, ...]
    # dp_inc[0] = [0, 0, ...] # dummy
    
    dp_val = [[None] * (n + 1) for _ in range(n + 1)]
    dp_inc = [[0] * (n + 1) for _ in range(n + 1)]
    
    dp_val[0][0] = Fraction(0, 1)
    
    for i in range(1, n + 1):
        d = i  # Denominator for item i-1 is i? 
        # Item index j corresponds to denominator j+2.
        # Item 0 -> denom 2.
        # Item i-1 -> denom i+1.
        # Wait, in the first loop:
        # for i in range(n): d = i + 2.
        # So item i (0-indexed) has denom i+2.
        # In the second loop, let's use idx from 0 to n-1.
        pass
        
    # Let's rewrite the second pass clearly.
    # idx: 0 to n-1
    # denom = idx + 2
    # val = 1/denom
    
    # dp_val[idx+1][k] depends on dp_val[idx][k] and dp_val[idx][k-1]
    
    for idx in range(n):
        denom = idx + 2
        val = Fraction(1, denom)
        
        for k in range(n + 1):
            # Option 1: Do not include item idx
            # Value comes from dp_val[idx][k]
            # Option 2: Include item idx
            # Value comes from dp_val[idx][k-1] + val, if valid (<1)
            
            best_v = None
            best_inc = 0
            
            # Check Option 1
            if dp_val[idx][k] is not None:
                best_v = dp_val[idx][k]
                best_inc = 0
                
            # Check Option 2
            if k > 0 and dp_val[idx][k-1] is not None:
                candidate = dp_val[idx][k-1] + val
                if candidate < 1:
                    if best_v is None or candidate > best_v:
                        best_v = candidate
                        best_inc = 1
                    # If equal, we can keep the previous choice or pick this one. 
                    # Picking 0 (not include) is usually safer for sparsity? 
                    # Doesn't matter for score.
            
            dp_val[idx+1][k] = best_v
            dp_inc[idx+1][k] = best_inc
            
    # Find the best k in the final row (idx = n-1, so row n)
    best_sum_final = None
    best_k_final = -1
    for k in range(n + 1):
        if dp_val[n][k] is not None:
            if best_sum_final is None or dp_val[n][k] > best_sum_final:
                best_sum_final = dp_val[n][k]
                best_k_final = k
            
    if best_k_final == -1:
        return []
        
    # Backtrack to find indices
    selected_indices = []
    k = best_k_final
    for idx in range(n - 1, -1, -1):
        if dp_inc[idx+1][k] == 1:
            selected_indices.append(idx)
            k -= 1
            
    selected_indices.reverse()
    return selected_indices
# EVOLVE-BLOCK-END
