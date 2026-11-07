"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors[p] = exp
    if n > 1:
        factors[n] = 1
    return factors

def construct():
    n = 256
    primes = _sieve_primes(n)
    
    # Compute prime factor dicts for each number in 2..n
    prime_factors = {}
    for i in range(2, n + 1):
        prime_factors[i] = _factorize(i, primes)
    
    def is_valid_set(S):
        """Check if set S satisfies the condition for all distinct a, b, c in S."""
        m = len(S)
        # To speed up, we can precompute or just check.
        # For m~54, m^3 is ~150k, which is fast.
        for i in range(m):
            a = S[i]
            for j in range(i + 1, m):
                b = S[j]
                for k in range(j + 1, m):
                    c = S[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Strategy 1: All primes
    S_primes = list(primes)
    
    # Strategy 2: Try to augment or swap
    # We know S_primes is valid.
    # Can we add any number from 2..256 not in S_primes?
    # If we add x, we must check validity.
    # If valid, we have a larger set.
    
    best_S = S_primes
    
    # Try adding each composite
    composites = [i for i in range(2, n + 1) if i not in S_primes]
    
    # Sort composites by number of prime factors (fewer is better for keeping set large)
    # Actually, if we add x, we might need to remove some primes.
    # Let's try simple addition first.
    for x in composites:
        new_S = S_primes + [x]
        if is_valid_set(new_S):
            best_S = new_S
            break # Found one, but maybe more?
            
    # If we found an improvement, try to add more
    # If not, try swaps
    # Given time constraints, just returning primes is safe.
    # But let's try a greedy construction that prioritizes primes, then composites.
    
    def can_add(a, S_list):
        S = S_list
        m = len(S)
        for i in range(m):
            for j in range(i + 1, m):
                b = S[i]
                c = S[j]
                if (b * c) % a == 0:
                    return False
        for i in range(m):
            b = S[i]
            for j in range(m):
                if j == i:
                    continue
                c = S[j]
                if (a * c) % b == 0:
                    return False
        return True

    # Greedy: Primes first, then composites sorted by "difficulty" (num prime factors)
    candidates = list(range(2, n + 1))
    
    def sort_key_greedy(x):
        pf = prime_factors[x]
        is_prime = len(pf) == 1
        # Primes first
        # Then fewer prime factors
        # Then larger value
        return (0 if is_prime else 1, len(pf), -x)
        
    c_greedy = sorted(candidates, key=sort_key_greedy)
    S_greedy = []
    for a in c_greedy:
        if can_add(a, S_greedy):
            S_greedy.append(a)
            
    if len(S_greedy) > len(best_S):
        best_S = S_greedy
        
    return list(best_S)
# EVOLVE-BLOCK-END
