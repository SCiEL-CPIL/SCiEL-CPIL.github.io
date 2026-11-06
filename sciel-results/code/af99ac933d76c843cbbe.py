"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors descending to prioritize larger values (helps minimize count)
    divisors.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of divisors[i:]
    suffix_sum = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    def dfs(start_idx, remaining_sum, count_needed, current_path):
        if count_needed == 0:
            return current_path if remaining_sum == 0 else None
        
        if remaining_sum < 0:
            return None
            
        # Pruning: if remaining_sum > sum of largest count_needed divisors available
        if start_idx + count_needed > len(divisors):
            return None
            
        max_possible = sum(divisors[start_idx : start_idx + count_needed])
        if remaining_sum > max_possible:
            return None
            
        # Pruning: if remaining_sum > suffix_sum[start_idx]
        if remaining_sum > suffix_sum[start_idx]:
            return None
            
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining_sum:
                continue
            # Check time
            if time.time() - start_time > DEADLINE:
                return None
                
            # Recurse
            result = dfs(i + 1, remaining_sum - d, count_needed - 1, current_path + [d])
            if result is not None:
                return result
                
        return None

    for k in range(1, len(divisors) + 1):
        # Optional: Prune k if sum of k largest < target
        if k > 0 and sum(divisors[:k]) < target:
            continue
            
        res = dfs(0, target, k, [])
        if res is not None:
            return res
            
    # Fallback: greedy (should not be reached for valid instances)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
# EVOLVE-BLOCK-END
