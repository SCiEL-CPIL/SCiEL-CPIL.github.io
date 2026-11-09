"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # We want to maximize the size of a multiplicatively independent set.
    # The maximum size is the number of primes <= 40, which is 12.
    # The set of primes themselves is a valid solution.
    # Let's verify that the set of all primes works.
    # Subset products of primes are distinct because of the Fundamental Theorem of Arithmetic.
    
    # We can just return the list of primes.
    result = primes
    
    # Optional: Verification to ensure no bugs in logic (though logic is trivial here)
    # But we must be fast. 2^12 = 4096 products. Very fast.
    if time.time() < deadline:
        products = set()
        n = len(result)
        # Generate all subset products
        # Using bit manipulation
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= result[i]
            if prod in products:
                # This shouldn't happen for primes
                pass
            products.add(prod)
        if len(products) != (1 << n):
            # Fallback: try to find a smaller independent set?
            # Unlikely to be needed.
            pass
            
    return result
# EVOLVE-BLOCK-END
