"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

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

def _try_build_random(k_target, n_max=40, deadline=None, rng=None):
    """Try to build a set of k_target distinct integers in 2..40 with distinct subset products."""
    if rng is None:
        rng = random.Random()
    
    candidates = list(range(2, n_max + 1))
    
    # Prefer numbers with fewer/smaller prime factors first (powers of 2, 3, 5, then primes)
    # Sort by number of distinct prime factors, then by value
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    candidates.sort(key=lambda x: (prime_factor_count(x), x))
    
    # Try multiple random orders with bias toward "safe" numbers
    for attempt in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        # Shuffle candidates with bias
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        
        # Put pure powers of 2 and 3 and small primes at front
        def priority(x):
            if x in [2, 4, 8, 16, 32]:
                return 0
            if x in [3, 9, 27]:
                return 1
            if x in [5, 25]:
                return 2
            if x in [7, 11, 13, 17, 19, 23, 29, 31, 37]:
                return 3
            return 4
        
        shuffled.sort(key=lambda x: (priority(x), rng.random()))
        
        selected = []
        products = {1}
        
        for num in shuffled:
            if len(selected) >= k_target:
                break
            if deadline is not None and time.time() > deadline:
                break
            
            # Check if adding num creates a collision
            new_products = set()
            for p in products:
                new_products.add(p * num)
            
            if new_products & products:
                continue
            
            selected.append(num)
            products |= new_products
        
        if len(selected) == k_target:
            # Final verification
            if _verify_subset_products(selected):
                return selected
    
    return None

def _greedy_build(k_target, n_max=40, deadline=None):
    """Greedy approach: add numbers that are "most independent"."""
    candidates = list(range(2, n_max + 1))
    selected = []
    products = {1}
    
    # Try to add numbers in order of increasing "danger"
    # Numbers that are powers of a single prime are safest
    for num in candidates:
        if len(selected) >= k_target:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        new_products = set()
        for p in products:
            new_products.add(p * num)
        
        if new_products & products:
            continue
        
        selected.append(num)
        products |= new_products
    
    if len(selected) == k_target:
        if _verify_subset_products(selected):
            return selected
    return None

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin
    
    # Try to find a set of 15 first (known to be achievable)
    best = None
    
    # Hardcoded known good set of 15
    known_15 = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Wait, that's 16 numbers. Let me recount.
    # 2, 4, 16 (3 numbers, powers of 2)
    # 3, 9 (2 numbers, powers of 3)
    # 5, 25 (2 numbers, powers of 5)
    # 7, 11, 13, 17, 19, 23, 29, 31, 37 (9 primes)
    # Total: 3+2+2+9 = 16 numbers
    
    # Let me verify this set
    if _verify_subset_products(known_15):
        best = known_15
    
    # Try to find a set of 16
    if best is None or len(best) < 16:
        # Try greedy
        result = _greedy_build(16, deadline=deadline)
        if result:
            best = result
    
    # Try to find a set of 17
    if best is None or len(best) < 17:
        rng = random.Random(42)
        result = _try_build_random(17, deadline=deadline, rng=rng)
        if result:
            best = result
    
    # Try to find a set of 18
    if best is None or len(best) < 18:
        rng = random.Random(123)
        result = _try_build_random(18, deadline=deadline, rng=rng)
        if result:
            best = result
    
    if best is None:
        # Fallback to primes
        from math import isqrt
        best = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Final verification
    if not _verify_subset_products(best):
        # This should never happen, but be safe
        from math import isqrt
        best = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Ensure all are Python ints in range 2..40
    result = []
    for x in best:
        if isinstance(x, bool):
            continue
        if not isinstance(x, int):
            continue
        if 2 <= x <= 40:
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
