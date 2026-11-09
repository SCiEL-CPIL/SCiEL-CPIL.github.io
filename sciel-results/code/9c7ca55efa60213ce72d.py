"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Add safe composites (prime powers with distinct subset-sum exponents)
    # Powers of 2: 2^1, 2^2, 2^4 -> exponents {1,2,4}, subset sums 0-7 all distinct
    # Powers of 3: 3^1, 3^2, 3^3 -> exponents {1,2,3}, subset sums 0-6 all distinct
    # Powers of 5: 5^1, 5^2 -> exponents {1,2}, subset sums 0-3 all distinct
    # Powers of 7: 7^1 only (49 > 40)
    
    composites = [4, 16, 9, 27, 25]
    
    result = primes + composites
    
    # Verify distinct subset products using bitmask enumeration
    # 17 items -> 2^17 = 131072 products
    def verify(nums):
        n = len(nums)
        seen = set()
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in seen:
                return False
            seen.add(prod)
        return True
    
    if time.time() > deadline:
        return result  # Return best found even if verification pending
    
    if not verify(result):
        # Fallback: remove 27 if it causes issues
        result_fallback = primes + [4, 16, 9, 25]
        if verify(result_fallback):
            return result_fallback
        return primes
        
    return result
# EVOLVE-BLOCK-END
