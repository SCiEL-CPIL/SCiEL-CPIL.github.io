import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)  # 479001600
    primes = [2, 3, 5, 7, 11]
    n_primes = len(primes)
    target_vec = (10, 5, 2, 1, 1)
    
    def from_vector(vec):
        result = 1
        for i in range(n_primes):
            if vec[i]:
                result *= primes[i] ** vec[i]
        return result
    
    def can_divide(rem, fac):
        for i in range(n_primes):
            if rem[i] < fac[i]:
                return False
        return True
    
    def sub_vec(rem, fac):
        return tuple(rem[i] - fac[i] for i in range(n_primes))
    
    # Generate all divisors > 12 of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute vectors
    divs_vec = []
    for d in valid_divs:
        n = d
        vec = [0] * n_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        divs_vec.append(tuple(vec))
    
    n_valid = len(valid_divs)
    
    best_max = target
    best_sol = [target]
    
    def dfs(start_idx, rem_vec, count, factors, max_allowed):
        nonlocal best_max, best_sol
        
        if time.time() - start_time > timeout:
            return
        
        if all(v == 0 for v in rem_vec):
            if count > 0:
                current_max = max(factors)
                if current_max < best_max:
                    best_max = current_max
                    best_sol = list(factors)
            return
        
        if count >= 12:
            return
        
        rem_prod = from_vector(rem_vec)
        slots = 12 - count
        
        min_factor = factors[-1] + 1 if count > 0 else 13
        max_factor = min(rem_prod, max_allowed)
        
        if min_factor > max_factor:
            return
        
        # Pruning: min product of slots distinct factors >= min_factor
        min_prod = 1
        for j in range(slots):
            min_prod *= (min_factor + j)
        if min_prod > rem_prod:
            return
        
        for i in range(start_idx, n_valid):
            c = valid_divs[i]
            if c > max_factor:
                break
            if c < min_factor:
                continue
            
            cv = divs_vec[i]
            if not can_divide(rem_vec, cv):
                continue
            
            if slots > 1:
                min_next_prod = 1
                for j in range(1, slots):
                    min_next_prod *= (c + j)
                if min_next_prod > rem_prod // c:
                    continue
            
            factors.append(c)
            new_rem = sub_vec(rem_vec, cv)
            dfs(i + 1, new_rem, count + 1, factors, max_allowed)
            factors.pop()
    
    # Try with a moderate initial bound to find a good solution quickly
    dfs(0, target_vec, 0, [], 500)
    
    # If we found something, try to improve with tighter bound
    if best_max < target and time.time() - start_time < timeout - 5:
        dfs(0, target_vec, 0, [], best_max - 1)
    
    # If still at target, try with larger bound
    if best_max == target and time.time() - start_time < timeout - 5:
        dfs(0, target_vec, 0, [], target // 2)
        if best_max < target and time.time() - start_time < timeout - 5:
            dfs(0, target_vec, 0, [], best_max - 1)
    
    return best_sol
