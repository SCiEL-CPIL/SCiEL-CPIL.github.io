import time
from math import isqrt

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    TARGET = 123457
    
    # Generate primes up to a reasonable limit
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # Generate primes up to TARGET (we only need primes up to TARGET/2 for factorization)
    primes_list = sieve(TARGET)
    
    def generate_smooth_numbers(limit, primes):
        """
        Generates all numbers <= limit that are composed of the given primes.
        Returns a set of such numbers.
        """
        smooths = set([1])
        frontier = [1]
        while frontier:
            x = frontier.pop()
            for p in primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooths:
                    smooths.add(nx)
                    frontier.append(nx)
        return smooths

    def largest_prime_factor(n):
        """Compute the largest prime factor of n."""
        if n <= 1:
            return 1
        original_n = n
        lp = 1
        for p in primes_list:
            if p * p > n:
                break
            while n % p == 0:
                lp = p
                n //= p
        if n > 1:
            lp = max(lp, n)
        return lp

    # Strategy: Binary search on the smoothness bound B
    # For a given B, check if there exists a pair (a, b) with a+b=TARGET
    # where both a and b are B-smooth (all prime factors <= B)
    
    def exists_smooth_pair(B, primes_up_to_B):
        """Check if there exists a pair (a, b) with a+b=TARGET, both B-smooth."""
        smooths = generate_smooth_numbers(TARGET, primes_up_to_B)
        for a in smooths:
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooths:
                return (a, b)
        return None

    # We'll iterate through primes in order, testing each as the bound B
    # This is more efficient than binary search because we can reuse the smooth number generation
    
    best_pair = None
    best_value = float('inf')
    
    # Start with small primes and expand
    current_primes = []
    for B in primes_list:
        if time.time() > deadline - 5:
            break
            
        current_primes.append(B)
        
        # Generate smooth numbers for these primes
        smooths = generate_smooth_numbers(TARGET, current_primes)
        
        # Check if any a in smooths has b = TARGET - a in smooths
        found = False
        for a in smooths:
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooths:
                if a > 0 and b > 0:
                    best_pair = (a, b)
                    best_value = B
                    found = True
                    break
        if found:
            break
            
    if best_pair:
        return [best_pair[0], best_pair[1]]
    else:
        # Fallback: compute max(P(a), P(b)) for several candidate pairs and pick the best
        # Try pairs where a and b are relatively small
        candidates = []
        for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 200, 500, 1000]:
            b = TARGET - a
            if b > 0:
                candidates.append((a, b))
        
        # Also try pairs around TARGET/2
        mid = TARGET // 2
        for offset in range(1, 1000):
            candidates.append((mid - offset, mid + 1 + offset))
            candidates.append((mid + offset, mid + 1 - offset))
        
        best_fallback = None
        best_fallback_val = float('inf')
        
        for a, b in candidates:
            if a <= 0 or b <= 0:
                continue
            pa = largest_prime_factor(a)
            pb = largest_prime_factor(b)
            val = max(pa, pb)
            if val < best_fallback_val:
                best_fallback_val = val
                best_fallback = (a, b)
        
        if best_fallback:
            return [best_fallback[0], best_fallback[1]]
        else:
            return [1, TARGET - 1]
