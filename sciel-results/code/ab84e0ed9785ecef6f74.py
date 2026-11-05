import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    # Cache for failed states: (rem_num, rem_den, min_den, terms_left) -> True (failed)
    failed_cache = set()
    
    def dfs(rem_num, rem_den, min_den, terms_left, current_denoms):
        # Check time limit
        if time.time() > deadline:
            return None
        
        # If no terms left, check if remaining fraction is 0
        if terms_left == 0:
            if rem_num == 0:
                return current_denoms
            return None
        
        # If one term left, it must be exactly the remaining fraction
        if terms_left == 1:
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= min_den and d <= 10000000:
                    return current_denoms + [d]
            return None

        # Check cache
        cache_key = (rem_num, rem_den, min_den, terms_left)
        if cache_key in failed_cache:
            return None

        # Calculate bounds for the next denominator d
        # 1/d <= rem_num/rem_den  => d >= rem_den/rem_num
        lower_bound = max(min_den, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound: 
        # The sum of terms_left unit fractions starting from d must be >= rem_num/rem_den
        # A tighter upper bound: since denominators must be distinct and increasing,
        # the maximum sum with terms_left terms starting at d is approximately
        # 1/d + 1/(d+1) + ... + 1/(d+terms_left-1)
        # For a loose bound: terms_left / d >= rem_num/rem_den => d <= terms_left * rem_den / rem_num
        upper_bound = min(10000000, int((terms_left * rem_den) // rem_num))
        
        # Additional pruning: check if even the best case can achieve the target
        # Max possible sum = sum of 1/(lower_bound + i) for i in range(terms_left)
        # This is an upper bound on what we can achieve
        if lower_bound > 10000000:
            failed_cache.add(cache_key)
            return None
        
        # If the maximum achievable sum is less than the target, fail
        # Approximate: terms_left / lower_bound should be >= rem_num/rem_den
        # More precise: sum_{i=0}^{terms_left-1} 1/(lower_bound+i) >= rem_num/rem_den
        # For speed, use the loose bound
        if terms_left * rem_den < rem_num * lower_bound:
            failed_cache.add(cache_key)
            return None
        
        # Iterate possible denominators
        # We prefer smaller denominators to keep the fraction "greedy" and potentially reduce depth
        for d in range(lower_bound, upper_bound + 1):
            if d > 10000000:
                break
            
            # New remaining fraction: rem_num/rem_den - 1/d
            # = (rem_num * d - rem_den) / (rem_den * d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            # new_num must be positive
            if new_num < 0:
                continue
            
            # Simplify fraction to keep numbers small
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Recurse
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current_denoms + [d])
            if result is not None:
                return result
                
            # Optional: Early exit if time is tight
            if time.time() > deadline:
                failed_cache.add(cache_key)
                return None

        failed_cache.add(cache_key)
        return None

    # Try increasing number of terms
    # k=1, 2 are impossible for d>=5
    for k in range(3, 20):
        result = dfs(1, 1, 5, k, [])
        if result is not None:
            return result
        
        if time.time() > deadline:
            break

    # Fallback: Greedy algorithm (Sylvester sequence variant)
    # This might produce more terms but is guaranteed to terminate quickly
    rem_num, rem_den = 1, 1
    denoms = []
    last_d = 4
    while rem_num > 0:
        if time.time() > deadline:
            break
        # Find smallest d > last_d such that 1/d <= rem_num/rem_den
        # d >= rem_den/rem_num
        d = max(last_d + 1, (rem_den + rem_num - 1) // rem_num)
        
        # Ensure d is within bounds
        if d > 10000000:
            break
            
        denoms.append(d)
        last_d = d
        
        # Update remainder
        # rem = rem - 1/d
        # new_num = rem_num * d - rem_den
        # new_den = rem_den * d
        new_num = rem_num * d - rem_den
        new_den = rem_den * d
        
        if new_num < 0:
            break
            
        g = math.gcd(new_num, new_den)
        rem_num = new_num // g
        rem_den = new_den // g
        
        # Safety check for infinite loops or overflow
        if len(denoms) > 256:
            break
            
    if rem_num == 0 and len(denoms) > 0:
        return denoms
        
    raise RuntimeError("no representation found")
