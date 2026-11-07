"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    # The maximum possible size is 16.
    # A known optimal set consists of:
    # Powers of 2: 2, 4, 16 (exponents 1, 2, 4)
    # Powers of 3: 3, 9 (exponents 1, 2)
    # Powers of 5: 5, 25 (exponents 1, 2)
    # Primes: 7, 11, 13, 17, 19, 23, 29, 31, 37
    best = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify to be safe
    if not _verify_subset_products(best):
        # Fallback to primes if the hardcoded set is somehow invalid (should not happen)
        from math import isqrt
        best = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
        
    # Ensure types and ranges
    result = []
    for x in best:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40:
            result.append(x)
            
    # Ensure distinct
    seen = set()
    unique = []
    for x in result:
        if x not in seen:
            seen.add(x)
            unique.append(x)
            
    return unique
# EVOLVE-BLOCK-END
