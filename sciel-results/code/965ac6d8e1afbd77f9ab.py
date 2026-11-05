import time
from math import isqrt

def _find_rep(p):
    """Find a, b such that a^2 + b^2 = p for prime p == 1 mod 4."""
    if p == 2:
        return (1, 1)
    # p is odd
    for a in range(1, isqrt(p) + 1):
        b2 = p - a * a
        b = isqrt(b2)
        if b * b == b2:
            return (a, b)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 10s safety margin
    
    # Generate primes p == 1 mod 4
    primes = []
    # We need to go up to maybe 1000?
    # Product of first 20 primes is large.
    # Let's just generate primes and filter.
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, isqrt(n) + 1):
            if n % i == 0:
                return False
        return True

    # Find enough primes
    candidate = 5
    while len(primes) < 30: # Safety cap
        if candidate % 4 == 1 and is_prime(candidate):
            primes.append(candidate)
        candidate += 2
        
    # Select subset of primes such that product <= 10^18
    N = 1
    selected_primes = []
    for p in primes:
        if N * p <= 10**18:
            N *= p
            selected_primes.append(p)
        else:
            break
            
    if not selected_primes:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    # Initialize representations
    # (x, y) such that x^2 + y^2 = current_product
    reps = {(1, 0)}
    
    for p in selected_primes:
        if time.time() > deadline - 10:
            break
        rep = _find_rep(p)
        if rep is None:
            continue
        a, b = rep
        new_reps = set()
        for x, y in reps:
            # (x + iy)(a + ib) = (xa - yb) + i(xb + ya)
            u1 = x * a - y * b
            v1 = x * b + y * a
            new_reps.add((abs(u1), abs(v1)))
            
            # (x + iy)(a - ib) = (xa + yb) + i(ya - xb)
            u2 = x * a + y * b
            v2 = y * a - x * b
            new_reps.add((abs(u2), abs(v2)))
        reps = new_reps
        
    # Extract pairs
    pairs = set()
    for x, y in reps:
        a_val = x * x
        b_val = y * y
        if a_val == 0 or b_val == 0:
            continue
        if a_val > b_val:
            a_val, b_val = b_val, a_val
        pairs.add((a_val, b_val))
        
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': N, 'pairs': [list(p) for p in pairs]}
