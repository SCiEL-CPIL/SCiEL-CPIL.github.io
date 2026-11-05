import math

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    # We only need primes up to N, but actually up to N/2 is enough for factors?
    # No, P(n) can be up to n.
    # But we are iterating K.
    # We need primes in ascending order.
    
    # Limit for primes: if we go up to N, it's safe.
    # Sieve size N is 123457, which is small.
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    smooths = {1}
    
    # Iterate through primes
    for p in primes:
        # Expand smooths with p
        # We iterate over a copy of the current smooths to avoid modification during iteration issues,
        # but we also need to consider newly added numbers in this step?
        # No, if we generate $x \cdot p^k$, we start from existing smooths $x$.
        # If we add $y = x \cdot p$, can we generate $y \cdot p = x \cdot p^2$?
        # Yes, if we iterate over the updated set or use a queue.
        # Using a list `current` that we update is tricky.
        # Let's use a simple approach:
        # For each x in current smooths, generate x*p, x*p^2, ...
        # We need to be careful not to re-process.
        
        # Actually, since we only add numbers that are multiples of p,
        # and we start from numbers that are NOT multiples of p (or are they?),
        # wait. Smooths for K-1 do not contain p.
        # So any new number must have p as a factor.
        # Let x be a (K-1)-smooth number.
        # New numbers are x * p^k.
        # We can just loop k=1, 2, ...
        
        new_numbers = []
        for x in list(smooths):
            val = x * p
            while val <= N:
                if val not in smooths:
                    new_numbers.append(val)
                val *= p
        
        smooths.update(new_numbers)
        
        # Check for solution
        # We can check during or after update.
        # Checking after is fine.
        for a in smooths:
            b = N - a
            if b in smooths:
                return [a, b]
                
    # Fallback (should not reach here if primes list is long enough)
    return [1, N - 1]
