import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80

    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    
    # Precompute suffix products for the full list?
    # No, the list changes with M.
    # But we can compute suffix products for the filtered list in each call.
    
    def search_for_M(M):
        valid = [d for d in all_divisors if d <= M]
        n = len(valid)
        
        if n == 0:
            return None
            
        # Compute suffix products
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * valid[i]
            
        # If total product of all valid divisors < T12, impossible
        if suffix_prod[0] < T12:
            return None

        def dfs(remaining, min_idx):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if min_idx >= n:
                return None
                
            # Pruning: if product of all remaining valid divisors is < remaining, fail
            if suffix_prod[min_idx] < remaining:
                return None
                
            # Pruning: if remaining is not divisible by any valid[i] for i >= min_idx, fail
            # We can check this in the loop.
            
            for i in range(min_idx, n):
                d = valid[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    res = dfs(remaining // d, i + 1)
                    if res is not None:
                        return [d] + res
                    # Optional: if we want to find *a* solution, we can stop here?
                    # No, we need to backtrack to find *any* solution.
                    # But if we find one, we return it.
            
            return None
        
        res = dfs(T12, 0)
        if res is not None:
            return res
        return None

    # Search M from 13 upwards
    # We know a solution exists with M <= sqrt(12!) ~ 21886
    # Let's search up to 25000
    for M in range(13, 25001):
        if time.time() > deadline:
            break
        sol = search_for_M(M)
        if sol is not None:
            return sol
            
    # Fallback
    return [T12]
