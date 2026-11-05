import time
from collections import deque

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 55  # Keep safety margin for 60s limit
    
    # Generate primes up to a reasonable limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(500)
    
    # For each prime p, generate all p-smooth numbers <= TARGET
    # Then check if there exists a pair a, b with a+b=TARGET and both are p-smooth
    # We want the smallest such p
    
    # Precompute smooth number sets incrementally
    # Start with just {1} and add primes one by one
    
    def generate_smooth_numbers(prime_list, limit):
        """Generate all numbers that are smooth with respect to prime_list and <= limit."""
        smooths = {1}
        for p in prime_list:
            new_smooths = set()
            for s in list(smooths):
                val = s
                while val <= limit:
                    new_smooths.add(val)
                    val *= p
            smooths = smooths | new_smooths
            smooths = {x for x in smooths if x <= limit}
        return smooths
    
    # Try increasing prime sets
    for num_primes in range(1, len(primes) + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        current_primes = primes[:num_primes]
        
        # Generate all smooth numbers
        smooths = generate_smooth_numbers(current_primes, TARGET)
        
        # Check for pair: a + b = TARGET, both in smooths
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                return [a, b]
    
    # Fallback: find a pair with small max prime factor by direct search
    best_a, best_b = 1, TARGET - 1
    best_max_p = float('inf')
    
    limit = min(TARGET // 2, 50000)
    for a in range(1, limit + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = TARGET - a
        # Compute largest prime factor of a
        def largest_prime_factor(n):
            if n <= 1:
                return 1
            f = 1
            d = 2
            while d * d <= n:
                while n % d == 0:
                    f = d
                    n //= d
                d += 1
            if n > 1:
                f = max(f, n)
            return f
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        mx = max(pa, pb)
        if mx < best_max_p:
            best_max_p = mx
            best_a, best_b = a, b
            if best_max_p == 2:  # Can't do better than 2
                return [best_a, best_b]
    
    return [best_a, best_b]
