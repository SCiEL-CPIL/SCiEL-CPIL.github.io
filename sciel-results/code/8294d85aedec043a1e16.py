import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    max_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        power = 1
        for _ in range(max_exponents[i] + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
        
    divisors.sort()
    
    # Precompute divisors of each divisor of N
    # divisors_of[d] = list of divisors of d that are also divisors of N
    # Since all divisors of d are divisors of N, we can just filter the global list
    # Or compute directly.
    # Given the small count (792), we can do this efficiently.
    divisors_set = set(divisors)
    divisors_of = {}
    for d in divisors:
        divs = []
        # Iterate through all divisors of N and check if they divide d?
        # No, iterate through divisors of N, if x divides d, add x.
        # This is 792 * 792 ~ 600k ops. Fast.
        for x in divisors:
            if d % x == 0:
                divs.append(x)
        divisors_of[d] = divs

    def can_decompose(remaining, min_factor, max_factor, depth):
        # remaining is the product of the remaining factors
        # min_factor is the minimum value for the next factor
        # max_factor is the maximum value allowed for any factor
        
        if time.time() > deadline:
            return False

        if remaining == 1:
            return True
        
        if depth >= 12:
            return False
            
        if min_factor > max_factor:
            return False
            
        # If remaining is prime and > max_factor, impossible
        # If remaining is not divisible by any number in [min_factor, max_factor], impossible
        
        candidates = divisors_of[remaining]
        
        # We need to find a divisor d of 'remaining' such that:
        # 1. min_factor <= d <= max_factor
        # 2. d != remaining (unless remaining is the last factor? 
        #    If we pick d=remaining, then remaining becomes 1.
        #    This is valid if d is in range.
        #    So d can be remaining.)
        
        # Optimization: 
        # If we pick d, the new remaining is R' = remaining // d.
        # The next factors must be > d.
        # So we need to be able to decompose R' into factors in (d, max_factor].
        
        # Iterate candidates in descending order? 
        # Picking larger d reduces remaining faster, potentially hitting 1 sooner.
        # But it constrains the next factors to be > d.
        # Let's try descending to find a solution quickly.
        
        # Filter candidates
        valid_candidates = [d for d in candidates if min_factor <= d <= max_factor]
        
        # If no candidates, return False
        if not valid_candidates:
            return False
            
        # Sort descending to try "big" splits first?
        # Actually, if we pick a very large d, R' becomes small.
        # If R' == 1, we are done.
        # If R' > 1, we need factors > d.
        # If d is close to max_factor, it's hard to find factors > d.
        # So maybe ascending is better? 
        # If we pick small d, R' is large, and we have more room for larger factors.
        # Let's try ascending.
        valid_candidates.sort()
        
        for d in valid_candidates:
            R_prime = remaining // d
            if R_prime == 1:
                return True
            
            # Check if R_prime can be decomposed into factors > d and <= max_factor
            # The smallest factor we can pick next is d + 1.
            # So we call can_decompose(R_prime, d + 1, max_factor, depth + 1)
            
            # Pruning: 
            # The product of the smallest possible distinct factors starting from d+1
            # must be <= R_prime? 
            # Actually, R_prime is the product of the remaining factors.
            # The minimum product of k distinct integers starting from d+1 is (d+1)...(d+k).
            # We don't know k.
            # But if R_prime < d + 1, it's impossible (unless R_prime==1 which is handled).
            if R_prime < d + 1:
                continue
                
            if can_decompose(R_prime, d + 1, max_factor, depth + 1):
                return True
                
        return False

    # Binary search for the minimum M
    # Lower bound: 13
    # Upper bound: N
    low = 13
    high = N
    best_M = N
    
    # We can narrow down the search space.
    # The product 13*14*...*13+k is roughly (13+k)!/12!
    # We want this to be >= 12! ? No, we want the product of factors to BE 12!.
    # So we need enough "capacity" in the range [13, M].
    # A very rough upper bound: 
    # If we take 12 factors, the smallest max is when they are as equal as possible.
    # 12! ^ (1/12) is approx 10. So 12 factors around 10 is not possible (must be > 12).
    # Let's just binary search.
    
    while low < high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        
        if can_decompose(N, 13, mid, 0):
            best_M = mid
            high = mid
        else:
            low = mid + 1
            
    # If we found best_M, we need to return the actual factors.
    # We need to reconstruct the solution for best_M.
    
    def find_factors(remaining, min_factor, max_factor, depth):
        if remaining == 1:
            return []
            
        if depth >= 12:
            return None
            
        if min_factor > max_factor:
            return None
            
        candidates = divisors_of[remaining]
        valid_candidates = [d for d in candidates if min_factor <= d <= max_factor]
        valid_candidates.sort()
        
        for d in valid_candidates:
            R_prime = remaining // d
            if R_prime == 1:
                return [d]
                
            if R_prime < d + 1:
                continue
                
            sub_factors = find_factors(R_prime, d + 1, max_factor, depth + 1)
            if sub_factors is not None:
                return [d] + sub_factors
                
        return None

    factors = find_factors(N, 13, best_M, 0)
    
    if factors is None:
        # Fallback to trivial if something went wrong (should not happen if can_decompose is consistent)
        return [N]
        
    return factors
