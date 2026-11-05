# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n_val = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisors.sort()
    n_divs = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    total_sum = prefix[n_divs]
    
    # We want to find the minimum k such that there exist k distinct divisors summing to target.
    # We iterate k from 1 upwards.
    
    for k in range(1, 257):
        if time.time() > deadline:
            # If we run out of time, we might not find the optimal k.
            # However, for this specific instance, k should be small.
            # If we reach here, we return an empty list or best effort?
            # The problem guarantees a solution exists.
            # If we time out, we score 0. So we must be fast.
            break
            
        # Check if it's possible to sum to target with k divisors
        # Min sum of k divisors is sum of smallest k: prefix[k]
        # Max sum of k divisors is sum of largest k: total_sum - prefix[n_divs - k]
        
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        
        if target < min_sum_k or target > max_sum_k:
            continue
            
        # Use backtracking with pruning
        chosen = []
        
        def backtrack(i, remaining_k, remaining_sum):
            # i is the current index in divisors
            # remaining_k is how many more divisors we need to pick
            # remaining_sum is the sum we need to achieve
            
            if remaining_k == 0:
                return remaining_sum == 0
            
            if i == n_divs:
                return False
            
            # Pruning 1: Not enough elements left
            if n_divs - i < remaining_k:
                return False
            
            # Pruning 2: Min possible sum of remaining_k elements from i
            # The smallest remaining_k elements are divisors[i : i+remaining_k]
            min_possible = prefix[i + remaining_k] - prefix[i]
            if remaining_sum < min_possible:
                return False
            
            # Pruning 3: Max possible sum of remaining_k elements from i
            # The largest remaining_k elements in divisors[i:] are the last remaining_k elements of the whole list
            # BUT we can only pick from i onwards.
            # If i <= n_divs - remaining_k, the largest possible sum is sum of the largest remaining_k divisors overall.
            # Otherwise, it is sum of all divisors from i to end.
            if i <= n_divs - remaining_k:
                max_possible = total_sum - prefix[n_divs - remaining_k]
            else:
                max_possible = total_sum - prefix[i]
            
            if remaining_sum > max_possible:
                return False
            
            # Branch 1: Pick divisors[i]
            # We must ensure that even if we pick the smallest possible for the rest, we don't go under?
            # The min_possible check above handles the case where we pick NOTHING from the rest? No.
            # The min_possible check is for the BEST case (smallest sum).
            # If remaining_sum < min_possible, it's impossible.
            
            # Let's try picking divisors[i]
            if backtrack(i + 1, remaining_k - 1, remaining_sum - divisors[i]):
                chosen.append(divisors[i])
                return True
                
            # Branch 2: Skip divisors[i]
            if backtrack(i + 1, remaining_k, remaining_sum):
                return True
                
            return False
        
        if backtrack(0, k, target):
            return chosen
            
    # Fallback (should not reach here if solution exists and time is sufficient)
    return []
# EVOLVE-BLOCK-END
