"""
Algorithm for dense increasing sequences with unique block products.

Strategy:
1. Iterate through integers from 2 to 128 in increasing order.
2. For each integer x, check if it can be appended to the current sequence
   such that all new block products are distinct from existing ones.
3. If valid, append x.
4. This greedy approach maximizes the number of elements by prioritizing
   smaller numbers which are less likely to cause large product collisions
   and allows more room for subsequent numbers.
"""
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    deadline = time.time() + 85.0  # Safety margin
    best_seq = []
    
    # We can try a few different greedy orders
    # 1. Increasing order (2, 3, ..., 128)
    # 2. Primes first, then composites
    # 3. Non-square-free composites, then primes, then square-free composites
    
    orders = []
    
    # Order 1: Natural increasing
    orders.append(list(range(2, 129)))
    
    # Order 2: Primes first
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    orders.append(primes + composites)
    
    # Order 3: Try to prioritize numbers that are "safe" (primes and non-square-free)
    # A number is non-square-free if it is divisible by p^2 for some prime p
    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True
        
    safe_composites = [x for x in composites if not is_square_free(x)]
    risky_composites = [x for x in composites if is_square_free(x)]
    orders.append(primes + safe_composites + risky_composites)

    for order in orders:
        if time.time() > deadline:
            break
            
        seq = []
        products = set()
        
        for x in order:
            if time.time() > deadline:
                break
                
            # Calculate new products if x is appended
            new_prods = set()
            p = x
            new_prods.add(p)
            
            for a in reversed(seq):
                p *= a
                new_prods.add(p)
                
            # Check for collisions
            if not new_prods.intersection(products):
                seq.append(x)
                products.update(new_prods)
                
        if len(seq) > len(best_seq):
            best_seq = seq
            
    return best_seq
