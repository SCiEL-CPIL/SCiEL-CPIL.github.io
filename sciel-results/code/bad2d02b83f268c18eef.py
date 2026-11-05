import time
import sys
from math import isqrt
from collections import Counter

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0/3.0)) + 2
    
    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i*i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue
        
        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Generate powerful numbers up to 50 million
    # This gives ~15000 numbers, K^2/2 ~ 112.5M operations
    # In CPython, simple addition + dict update is ~50-100ns, so ~6-12 seconds
    pow_limit = 50_000_000
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # If generation took too long, reduce limit and retry
    if time.time() > deadline - 60:
        pow_limit = 20_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)
    
    # Count all pair sums using a fast inner loop
    counts = {}
    pow_list = powerful
    local_time = time.time
    local_deadline = deadline - 10
    
    # Use local variable binding for speed
    for i in range(K):
        a = pow_list[i]
        for j in range(i, K):
            s = a + pow_list[j]
            counts[s] = counts.get(s, 0) + 1
    
    # Check if we need to bail out early
    if local_time() > local_deadline and not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    # Find the best N (highest count)
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Extract all pairs for this N
    pow_set = set(pow_list)
    pairs = []
    for a in pow_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
    
    # Verify each pair
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
    
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
    
    return {'N': best_N, 'pairs': verified_pairs}
