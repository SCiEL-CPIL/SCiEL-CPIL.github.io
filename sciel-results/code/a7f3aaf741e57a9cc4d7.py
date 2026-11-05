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
    
    # k=3: O(n^2) using set with tighter bounds
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        rem2 = target - a
        # Need two distinct divisors > a that sum to rem2
        # Each must be > a and < rem2
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            complement = target - a - b
            if complement in divisor_set and complement > b:
                return [a, b, complement]
    
    # For k >= 4, use optimized backtracking
    # Sort in descending order for tighter max-sum pruning
    divs_desc = sorted(divisors, reverse=True)
    n = n_divs
    
    # Suffix sums: suffix[i] = sum of divs_desc[i:]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + divs_desc[i]
    
    # Memoization set for infeasible states
    memo = set()
    
    def search(idx, k_rem, sum_rem, chosen):
        if k_rem == 0:
            return sum_rem == 0
        
        if idx >= n:
            return False
        
        if n - idx < k_rem:
            return False
        
        # Time check
        if time.time() > deadline:
            return False
        
        state = (idx, k_rem, sum_rem)
        if state in memo:
            return False
        
        # Min possible sum: k_rem smallest elements available from idx onwards
        # In descending order, the smallest k_rem elements overall are at indices n-k_rem to n-1
        # We can use them if idx <= n - k_rem
        min_possible = total_sum - prefix[n - k_rem]
        if sum_rem < min_possible:
            memo.add(state)
            return False
        
        # Max possible sum: k_rem largest elements from idx onwards = divs_desc[idx:idx+k_rem]
        max_possible = suffix[idx] - suffix[idx + k_rem]
        if sum_rem > max_possible:
            memo.add(state)
            return False
        
        # Try picking divs_desc[idx]
        if divs_desc[idx] <= sum_rem:
            chosen.append(divs_desc[idx])
            if search(idx + 1, k_rem - 1, sum_rem - divs_desc[idx], chosen):
                return True
            chosen.pop()
        
        # Try skipping divs_desc[idx]
        if search(idx + 1, k_rem, sum_rem, chosen):
            return True
        
        memo.add(state)
        return False
    
    for k in range(4, 257):
        if time.time() > deadline:
            break
        
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        
        if target < min_sum_k or target > max_sum_k:
            continue
        
        memo.clear()
        chosen = []
        
        if search(0, k, target, chosen):
            return chosen
    
    # Fallback: greedy
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
