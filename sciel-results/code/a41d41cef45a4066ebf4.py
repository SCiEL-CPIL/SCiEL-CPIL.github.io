"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    S = [2, 3, 5, 7, 11, 13]
    
    # Identify primes in range
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, isqrt(n) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [i for i in range(2, N + 1) if is_prime(i)]
    
    # Primes >= 17
    B = [p for p in primes if p >= 17]
    
    # Numbers in A: divisible by at least one prime in S
    A = []
    for i in range(2, N + 1):
        for s in S:
            if i % s == 0:
                A.append(i)
                break
                
    # Precompute masks for A
    # Mask bit i corresponds to S[i]
    a_masks = []
    for a in A:
        mask = 0
        for i, s in enumerate(S):
            if a % s == 0:
                mask |= (1 << i)
        a_masks.append(mask)
        
    # DP for A
    # dp[mask] = max sum
    dp = [0] * (1 << len(S))
    
    # We want to pick a subset of A with disjoint masks
    # Iterate through items
    # To be safe with time, we just do standard 0/1 knapsack logic over masks
    # Since we can only pick one item per "mask group" effectively?
    # No, we can pick multiple items if masks are disjoint.
    # Standard DP:
    # For each item (a, m):
    #   for mask in reversed(range(64)):
    #     if mask & m == 0:
    #       dp[mask | m] = max(dp[mask | m], dp[mask] + a)
       
    # Optimization: Group items by mask?
    # If two items have same mask, they conflict. Keep only max value?
    # Yes.
    best_val_for_mask = [0] * (1 << len(S))
    for i, a in enumerate(A):
        m = a_masks[i]
        if a > best_val_for_mask[m]:
            best_val_for_mask[m] = a
            
    # Now we have at most 64 items (one per mask)
    # Apply DP
    for m in range(1 << len(S)):
        val = best_val_for_mask[m]
        if val == 0:
            continue
        # Iterate masks backwards to avoid using same item twice?
        # Actually, since we reduced to 1 item per mask, we can just iterate all masks
        # But we must ensure we don't combine an item with itself?
        # The mask check `mask & m == 0` prevents using an item if its mask is already in `mask`?
        # No, `mask` is the union of PREVIOUSLY selected items.
        # If we select item with mask `m`, new mask is `mask | m`.
        # We need `mask & m == 0`.
        for curr_mask in range(1 << len(S)):
            if curr_mask & m == 0:
                new_mask = curr_mask | m
                if dp[curr_mask] + val > dp[new_mask]:
                    dp[new_mask] = dp[curr_mask] + val
                    
    max_a_sum = max(dp)
    
    result = [1] + B
    # We don't need to reconstruct A for the sum, just the value is needed for the score?
    # No, we need to return the list.
    # But wait, the task says "Return a list of ints".
    # And "The evaluator measures the raw value (higher is better)".
    # So we need to return the actual numbers.
    # We need to reconstruct the optimal subset of A.
    
    # Reconstruct A subset
    # We need to store the choice made in DP.
    # Since it's small, we can re-run DP with backtracking or store parent pointers.
    # Given the small size, let's just store the items chosen.
    
    # Re-do DP with tracking
    dp_val = [0] * (1 << len(S))
    dp_parent = [None] * (1 << len(S)) # (prev_mask, item_mask, item_val)
    
    for m in range(1 << len(S)):
        val = best_val_for_mask[m]
        if val == 0:
            continue
        for curr_mask in range(1 << len(S)):
            if curr_mask & m == 0:
                new_mask = curr_mask | m
                if dp_val[curr_mask] + val > dp_val[new_mask]:
                    dp_val[new_mask] = dp_val[curr_mask] + val
                    dp_parent[new_mask] = (curr_mask, m, val)
                    
    # Find best mask
    best_mask = 0
    best_sum = 0
    for m in range(1 << len(S)):
        if dp_val[m] > best_sum:
            best_sum = dp_val[m]
            best_mask = m
            
    # Backtrack
    selected_a = []
    curr = best_mask
    while curr != 0:
        parent = dp_parent[curr]
        if parent is None:
            break
        prev_mask, m, val = parent
        # Find the actual number with mask m and value val
        # There might be multiple, but we picked the max one in best_val_for_mask
        # So we just need to find a in A with a_masks[i]==m and a==val
        for i, a in enumerate(A):
            if a_masks[i] == m and a == val:
                selected_a.append(a)
                break
        curr = prev_mask
        
    result.extend(selected_a)
    
    return result
# EVOLVE-BLOCK-END
