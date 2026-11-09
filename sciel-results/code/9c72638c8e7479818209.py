"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Preprocess candidates
    # We want to maximize sum 1/(512 - a).
    # This means we prefer large 'a' (close to 511).
    # However, we are constrained by pairwise coprimality.
    # Let's sort candidates by "value" descending: 1/(512-a) is decreasing in (512-a), so increasing in a.
    # So we sort 'a' from 511 down to 1.
    
    candidates = list(range(511, 0, -1))
    
    # Precompute prime factors for each candidate to speed up GCD checks?
    # Or just use math.gcd which is fast.
    # Let's use a set of chosen numbers and check gcd against all.
    
    best_sum = 0.0
    best_set = set()
    
    # Branch and Bound
    # State: index in candidates, current chosen set, current sum, current product of primes? No, just set.
    # To prune: sum_current + sum(1/(512-candidates[i:]) for all i >= idx) <= best_sum
    
    # Precompute suffix sums of values for the bound
    # val[i] = 1 / (512 - candidates[i])
    vals = [1.0 / (512 - a) for a in candidates]
    n = len(candidates)
    
    # suffix_sum[i] = sum(vals[i:])
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + vals[i]
        
    # Recursive DFS
    def dfs(idx, chosen, current_sum):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best if current is better
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(chosen)
            
        # Bound check
        # If current_sum + sum of all remaining values <= best_sum, prune
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Iterate through candidates from idx to end
        for i in range(idx, n):
            a = candidates[i]
            
            # Check coprimality
            valid = True
            for b in chosen:
                if math.gcd(a, b) != 1:
                    valid = False
                    break
            
            if valid:
                chosen.append(a)
                dfs(i + 1, chosen, current_sum + vals[i])
                chosen.pop()
                
            # Time check inside loop to avoid deep recursion timeout
            if time.time() > deadline:
                return

    dfs(0, [], 0.0)
    
    # Return the list of integers
    return list(best_set)
# EVOLVE-BLOCK-END
