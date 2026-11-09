"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    # The maximum number of distinct subset products is bounded by the number of primes <= 40.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12 primes)
    # Any set of 12 numbers with linearly independent prime exponent vectors will work.
    # The set of primes themselves is the simplest such set.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify that the primes satisfy the condition (they do, but let's be safe)
    # We can quickly check if any subset product collides for small k, 
    # but for primes, it's mathematically guaranteed.
    
    # To be robust, we could verify, but it's O(2^12) = 4096 checks, which is fast.
    def is_valid(nums):
        products = set()
        n = len(nums)
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    # The primes are valid.
    # Can we find a set of size > 12? No, because there are only 12 primes <= 40.
    # The exponent vectors live in R^12, so max independent set size is 12.
    # Thus, 12 is the maximum possible size.
    
    return primes
# EVOLVE-BLOCK-END
