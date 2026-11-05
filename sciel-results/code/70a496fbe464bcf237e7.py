import time
from math import isqrt

def construct():
    start_time = time.time()
    target = 479001600 # 12!
    
    # Precompute divisors of target that are > 12
    divisors = []
    for i in range(1, isqrt(target) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    divisors.sort()
    
    # Map each number to its divisors for quick lookup? 
    # Or just filter the global list. Since target is fixed, we can precompute
    # for each divisor d of target, the list of divisors of d that are > 12.
    # This might be heavy. Let's just filter the global sorted list.
    
    # Actually, iterating over all divisors of `rem` is better.
    # We can generate divisors of `rem` on the fly or precompute for all divisors of target.
    # There are not too many divisors. 12! = 2^10 * 3^5 * 5^2 * 7 * 11.
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # This is small. We can precompute divisors for each divisor of target.
    
    # Let's build a dictionary: divs_of[x] = sorted list of divisors of x that are > 12
    # We only need this for x that are divisors of target.
    
    all_divs_of_target = set(divisors)
    all_divs_of_target.add(1) # 1 is a divisor, but we don't use it as a factor > 12
    
    # We need divisors of any sub-product `rem`. `rem` will always be a divisor of target.
    divs_cache = {}
    for d in all_divs_of_target:
        if d == 1:
            divs_cache[1] = []
            continue
        ds = []
        # Find divisors of d
        for i in range(1, isqrt(d) + 1):
            if d % i == 0:
                if i > 12:
                    ds.append(i)
                other = d // i
                if other != i and other > 12:
                    ds.append(other)
        ds.sort()
        divs_cache[d] = ds

    def dfs(rem, last, count, path):
        nonlocal start_time
        if time.time() - start_time > 50: # Safety margin
            return None
        
        if rem == 1:
            return path[:]
        
        if count >= 12:
            return None
            
        # Pruning: 
        # If rem is prime, it must be the last factor.
        # Check if rem is prime and last < rem <= M
        # Actually, the loop handles this.
        
        # Get candidates: divisors of rem that are > last and <= M
        candidates = divs_cache[rem]
        # Filter candidates
        # Since candidates is sorted, we can use binary search or just iterate.
        # The list is small (< 100 usually).
        
        for c in candidates:
            if c <= last:
                continue
            if c > M:
                break # Since sorted, all subsequent are > M
            
            path.append(c)
            res = dfs(rem // c, c, count + 1, path)
            if res is not None:
                return res
            path.pop()
                
        return None

    M = 13
    while M <= 100: # Upper bound safety
        if time.time() - start_time > 55:
            break
        path = []
        res = dfs(target, 12, 0, path)
        if res is not None:
            return res
        M += 1
        
    # Fallback if not found (should not happen)
    return [target]
