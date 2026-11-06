import time
from math import log, isqrt
from collections import defaultdict

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    N = 123457
    
    # Precompute primes up to some limit
    # We'll generate smooth numbers for increasing t
    # First, get all primes up to N (we'll need them for checking P(n))
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Function to compute largest prime factor
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        orig = n
        p = 1
        for prime in primes:
            if prime > isqrt(n) and prime > n:
                break
            if prime * prime > n:
                break
            if n % prime == 0:
                p = prime
                while n % prime == 0:
                    n //= prime
        if n > 1:
            p = n
        return p
    
    # Generate t-smooth numbers up to limit
    def generate_smooth(primes_list, limit):
        """Generate all numbers ≤ limit whose prime factors are all in primes_list."""
        if not primes_list:
            return [1] if limit >= 1 else []
        results = {1}
        for p in primes_list:
            new_results = set()
            for val in results:
                mult = val
                while mult * p <= limit:
                    mult *= p
                    new_results.add(mult)
            results |= new_results
        return list(results)
    
    best = (1, 123456)
    best_val = largest_prime_factor(123456)  # = 149
    
    # Try increasing t
    for t in range(2, 200):
        if time.time() > deadline - 5:
            break
        
        # Get primes ≤ t
        small_primes = [p for p in primes if p <= t]
        
        smooth_nums = generate_smooth(small_primes, N)
        smooth_set = set(smooth_nums)
        
        # Check for valid pairs
        found = False
        for a in smooth_nums:
            if time.time() > deadline - 3:
                break
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                pa = largest_prime_factor(a)
                pb = largest_prime_factor(b)
                val = max(pa, pb)
                if val < best_val:
                    best_val = val
                    best = (a, b)
                found = True
                break  # Found a pair with this t, move to next t
        
        if found:
            # We found a pair with max P ≤ t. Keep going to see if smaller t works
            # Actually we're going in increasing t order, so first found is smallest t
            # But we want to minimize, so keep going? No - smaller t means smaller max P
            # Since we go t=2,3,4,..., the first t where we find a pair gives the smallest possible max P
            # Wait no - we should stop at the first t where we find a pair, because that's the minimum
            # Actually no: for t=50, we might find a pair, but for t=40 we might not have checked yet
            # Since we go in increasing order, the first t that works is the best
            break
    
    return [best[0], best[1]]
