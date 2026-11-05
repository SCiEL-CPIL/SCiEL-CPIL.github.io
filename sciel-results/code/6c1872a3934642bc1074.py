"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    n_val = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisor_set = set(divisors)
    divisors.sort()
    n_divs = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    total_sum = prefix[n_divs]
    
    # k=1: check if target itself is a divisor
    if target in divisor_set:
        return [target]
    
    # k=2: O(n) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement != d and complement > 0:
            return [d, complement]
    
    # k=3: O(n^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            complement = target - a - b
            if complement in divisor_set and complement > b:
                return [a, b, complement]
    
    # For k >= 4, use backtracking with pruning
    for k in range(4, 257):
        if time.time() > deadline:
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
            
    # Fallback: greedy (should not reach here if solution exists and time is sufficient)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    
    return chosen
# EVOLVE-BLOCK-END
