import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    target = factorial(12)
    
    # Generate all divisors of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    # We want to minimize the largest factor.
    # Let's iterate the upper bound B for the largest factor.
    # The smallest possible factor is 13.
    # The largest possible factor in an optimal solution is likely small.
    # We can iterate B from 13 up to max(valid_divs).
    
    # To speed up, we can just iterate through valid_divs as potential B values.
    # If a solution exists with max factor B, it must use only divisors <= B.
    
    for i in range(len(valid_divs)):
        B = valid_divs[i]
        if time.time() > deadline:
            # Fallback: return a known valid solution, e.g., [target]
            return [target]
            
        # Candidates are valid_divs[0..i]
        cands = valid_divs[:i+1]
        
        # Precompute suffix products for pruning
        # suffix_prod[j] = product(cands[j:])
        n_cands = len(cands)
        suffix_prod = [1] * (n_cands + 1)
        for j in range(n_cands - 1, -1, -1):
            suffix_prod[j] = suffix_prod[j+1] * cands[j]
            
        def dfs(idx, prod, last_factor, current_path):
            if time.time() > deadline:
                return None
            
            if prod == target:
                return list(current_path)
            
            if idx >= n_cands:
                return None
                
            # Pruning: If current product times the product of all remaining candidates
            # is less than target, we can't reach target.
            if prod * suffix_prod[idx] < target:
                return None
                
            # Option 1: Skip cands[idx]
            res = dfs(idx + 1, prod, last_factor, current_path)
            if res is not None:
                return res
                
            # Option 2: Take cands[idx]
            # Must be distinct: cands[idx] > last_factor
            # Must divide target/prod
            d = cands[idx]
            if d > last_factor:
                if target % (prod * d) == 0:
                    current_path.append(d)
                    res = dfs(idx + 1, prod * d, d, current_path)
                    if res is not None:
                        return res
                    current_path.pop()
                    
            return None
            
        solution = dfs(0, 1, 12, [])
        if solution is not None:
            return solution
            
    # If loop finishes without finding (should not happen), return fallback
    return [target]
