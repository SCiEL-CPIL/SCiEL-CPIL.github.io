import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    # 1. Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # 2. Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Helper to check if a number is prime
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Precompute primality for all valid divisors
    prime_status = {d: is_prime(d) for d in valid_divs}
    prime_status[1] = False
    
    n_valid = len(valid_divs)
    
    # Iterate M from smallest to largest
    for M_idx in range(n_valid):
        M = valid_divs[M_idx]
        
        def dfs(start_idx, remaining, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if remaining == 1:
                if len(current_factors) > 0:
                    return list(current_factors)
                return None
            
            if len(current_factors) >= 12:
                return None
                
            # If remaining is prime, it must be a single factor
            if prime_status.get(remaining, False):
                # Check if remaining is in valid_divs and index >= start_idx and <= M_idx
                # Since remaining is a divisor of target, and target's divisors are in valid_divs (if >12)
                # If remaining <= 12, it's not in valid_divs, so prime_status won't have it (or it's False if we checked).
                # Actually, if remaining is prime and > 12, it must be in valid_divs.
                # We need to check if its index is within [start_idx, M_idx].
                # We can use a dictionary for index lookup to be fast.
                # But we can also just check if remaining <= M and remaining >= valid_divs[start_idx] (approx)
                # Better: maintain a dict.
                pass # Placeholder, will implement below
            
            # Pruning: if the smallest available candidate is > remaining, fail
            if start_idx > M_idx:
                return None
            if valid_divs[start_idx] > remaining:
                return None
                
            # Iterate candidates
            for i in range(start_idx, M_idx + 1):
                c = valid_divs[i]
                
                if c > remaining:
                    break
                
                if remaining % c == 0:
                    current_factors.append(c)
                    res = dfs(i + 1, remaining // c, current_factors)
                    if res:
                        current_factors.pop()
                        return res
                    current_factors.pop()
                
            return None
        
        # We need index lookup for the prime check optimization
        # Let's create it outside the loop if it's static, or inside if it depends on M? 
        # The indices are static for valid_divs.
        div_index = {d: i for i, d in enumerate(valid_divs)}
        
        # Redefine dfs to use div_index
        def dfs_opt(start_idx, remaining, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if remaining == 1:
                if len(current_factors) > 0:
                    return list(current_factors)
                return None
            
            if len(current_factors) >= 12:
                return None
            
            # If remaining is prime
            if prime_status.get(remaining, False):
                if remaining in div_index:
                    idx = div_index[remaining]
                    if start_idx <= idx <= M_idx:
                        current_factors.append(remaining)
                        return list(current_factors)
                return None
            
            # Pruning: if the smallest available candidate is > remaining, fail
            if start_idx > M_idx:
                return None
            if valid_divs[start_idx] > remaining:
                return None
                
            # Iterate candidates
            for i in range(start_idx, M_idx + 1):
                c = valid_divs[i]
                
                if c > remaining:
                    break
                
                if remaining % c == 0:
                    current_factors.append(c)
                    res = dfs_opt(i + 1, remaining // c, current_factors)
                    if res:
                        current_factors.pop()
                        return res
                    current_factors.pop()
                
            return None

        sol = dfs_opt(0, target, [])
        
        if sol:
            return sol
            
        if time.time() - start_time > timeout:
            break

    return [target]
