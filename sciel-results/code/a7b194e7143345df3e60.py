import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # We want to find the largest subset of {2, ..., 40} with distinct subset products.
    # This is equivalent to finding the maximum number of multiplicatively independent integers.
    # This is bounded by the number of primes <= 40.
    
    # Primes <= 40
    primes = []
    for i in range(2, 41):
        is_prime = True
        for d in range(2, int(math.isqrt(i)) + 1):
            if i % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    # The set of all primes <= 40 is multiplicatively independent.
    # Its size is 12.
    # We can try to see if we can fit more by replacing primes with composites?
    # No, the rank is fixed at 12. We cannot exceed 12.
    # So returning all 12 primes is optimal.
    
    candidate = primes
    
    # Verification step to ensure validity (distinct subset products)
    # This is fast for k=12 (2^12 = 4096 checks)
    def check_distinct_subset_products(nums):
        k = len(nums)
        products = set()
        # Iterate over all subsets
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    if not check_distinct_subset_products(candidate):
        # Fallback: remove the largest prime? Or just return empty?
        # This should never happen for primes.
        # But just in case, we try to find a valid subset greedily if the full set fails.
        # However, primes always work.
        pass

    # Ensure types are correct
    result = [int(x) for x in candidate]
    
    # Check deadline
    if time.time() > deadline:
        # Return best found so far
        return result
        
    return result
