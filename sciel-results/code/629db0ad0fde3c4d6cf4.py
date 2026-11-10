"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # The set of integers is constructed from prime powers.
    # For each prime p, we select a set of exponents E_p such that
    # all subset sums of E_p are distinct.
    # This ensures that the combined set of numbers has distinct subset products.
    
    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Max exponent for each prime such that p^e <= 40
    # 2: 1, 2, 3, 4, 5 (32)
    # 3: 1, 2, 3 (27)
    # 5: 1, 2 (25)
    # 7: 1 (7)
    # others: 1
    
    # We choose exponents to maximize count while keeping subset sums distinct.
    # For p=2, max 3 exponents: {1, 2, 4} -> sums 0..7
    # For p=3, max 2 exponents: {1, 2} -> sums 0..3
    # For p=5, max 2 exponents: {1, 2} -> sums 0..3
    # For others, 1 exponent: {1} -> sums 0..1
    
    result = []
    
    # Add powers of 2
    result.extend([2**1, 2**2, 2**4]) # 2, 4, 16
    
    # Add powers of 3
    result.extend([3**1, 3**2]) # 3, 9
    
    # Add powers of 5
    result.extend([5**1, 5**2]) # 5, 25
    
    # Add powers of 7
    result.append(7**1) # 7
    
    # Add remaining primes
    for p in [11, 13, 17, 19, 23, 29, 31, 37]:
        result.append(p)
        
    # Verify distinctness and range
    if len(set(result)) != len(result):
        return []
    if any(x < 2 or x > 40 for x in result):
        return []
        
    # Optional: Verify subset products distinctness (expensive for 2^16, but doable)
    # 2^16 = 65536 products.
    # We can do a quick check if time permits, but the construction is theoretically sound.
    # Given the 60s limit, we skip full verification unless we are unsure.
    # However, to be safe against any logical error in my derivation:
    
    # Let's do a fast check for collisions using a set of products.
    # 65536 iterations is very fast in Python.
    products = set()
    n = len(result)
    for mask in range(1 << n):
        prod = 1
        # Calculate product
        # Optimization: only multiply if bit is set
        # 16 bits, so 16 checks per mask. 65536 * 16 ~ 1M ops. Fast.
        for i in range(n):
            if mask & (1 << i):
                prod *= result[i]
        if prod in products:
            return [] # Collision found, fallback to primes
        products.add(prod)
        
    return result

# EVOLVE-BLOCK-END
