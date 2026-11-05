import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, keep 5s buffer
    
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Generate all divisors of target that are > 12
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    divisors = sorted(divisors)
    n_divs = len(divisors)
    
    # For each possible max M (divisors[j]), check if we can form target
    # using distinct divisors all <= M
    # Use backtracking with memoization
    
    best_result = None
    
    for j in range(n_divs):
        M = divisors[j]
        
        # Get all divisors <= M that are > 12
        candidates = [d for d in divisors if d <= M]
        n_cand = len(candidates)
        
        if n_cand < 1:
            continue
        
        # Backtracking search
        # We need to pick a subset of candidates whose product is target
        # Use recursive search with pruning
        
        memo = {}
        
        def search(rem, start_idx):
            if rem == 1:
                return ()
            
            key = (rem, start_idx)
            if key in memo:
                return memo[key]
            
            # Pruning: if product of all remaining candidates < rem, impossible
            # (but this is expensive to compute, skip for now)
            
            for i in range(start_idx, n_cand):
                d = candidates[i]
                if rem % d == 0:
                    if time.time() > deadline:
                        memo[key] = None
                        return None
                    res = search(rem // d, i + 1)
                    if res is not None:
                        memo[key] = (d,) + res
                        return memo[key]
            
            memo[key] = None
            return None
        
        res = search(target, 0)
        
        if res is not None:
            best_result = sorted(res)
            break
        
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: return target itself
        return [target]
