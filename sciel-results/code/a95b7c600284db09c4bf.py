import time
import math

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    divisors = [1]
    for i, p in enumerate(primes):
        max_exp = target_exponents[i]
        new_divs = []
        for d in divisors:
            for e in range(max_exp + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    
    # Filter divisors > 12
    valid_candidates = []
    for d in divisors:
        if d > 12:
            # Compute exponent vector
            exps = [0] * 5
            temp = d
            for i, p in enumerate(primes):
                while temp % p == 0:
                    exps[i] += 1
                    temp //= p
            # Since d is a divisor of 12!, temp must be 1
            valid_candidates.append((d, tuple(exps)))
            
    # Sort candidates by value to allow early termination in DFS
    valid_candidates.sort(key=lambda x: x[0])
    
    best_max_val = float('inf')
    best_factors = []
    
    # To provide an initial bound, we can try a few simple heuristics or just start with inf
    # A simple bound: pick 2 factors. a*b = 12!. min(a,b) > 12.
    # Just let the DFS find it.
    
    def dfs(start_idx, current_exps, current_path):
        nonlocal best_max_val, best_factors
        
        # Time check
        if time.time() - start_time > 85:
            return

        # Check if we found a solution
        if current_exps == tuple(target_exponents):
            if current_path:
                current_max = current_path[-1] # Since we add in sorted order, last is max
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(current_path)
            return

        # If we have more than 12 factors, stop (though unlikely to reach this depth with product constraint)
        if len(current_path) >= 12:
            return

        for i in range(start_idx, len(valid_candidates)):
            x, exps = valid_candidates[i]
            
            # Pruning 1: If x >= best_max_val, no need to check this or any further (larger) candidates
            # because any solution using x will have max >= x >= best_max_val
            if x >= best_max_val:
                break

            # Pruning 2: Check if adding x exceeds target exponents
            can_add = True
            for j in range(5):
                if current_exps[j] + exps[j] > target_exponents[j]:
                    can_add = False
                    break
            if not can_add:
                continue

            # Additional Pruning: 
            # The product of remaining factors must be >= 1 (trivial)
            # But we can check if the remaining exponents can be formed by *any* combination.
            # A simple check: the smallest remaining candidate must be <= the "value" of remaining exponents?
            # This is hard to compute quickly.
            
            # Let's just recurse
            new_exps = tuple(current_exps[j] + exps[j] for j in range(5))
            current_path.append(x)
            dfs(i + 1, new_exps, current_path)
            current_path.pop()

    dfs(0, (0, 0, 0, 0, 0), [])
    
    # Fallback if no solution found (should not happen, but [N] is a valid solution if N>12)
    # Wait, N=12! is a divisor > 12. It has exponents (10,5,2,1,1).
    # So dfs should find [N] at least.
    if not best_factors:
        best_factors = [N]
        
    return best_factors
