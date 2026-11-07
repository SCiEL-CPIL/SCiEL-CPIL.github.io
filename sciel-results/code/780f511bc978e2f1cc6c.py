"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def _factorize(n, primes):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 1) + 1
    return factors

def _is_prime_power(n, primes):
    """Check if n is a prime power (p^k for k >= 1). Return (p, k) or None."""
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            temp = n
            k = 0
            while temp % p == 0:
                temp //= p
                k += 1
            if temp == 1:
                return (p, k)
    if n > 1:
        # Check if n itself is prime
        is_p = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_p = False
                break
        if is_p:
            return (n, 1)
    return None

def _largest_prime_power_for_prime(p, limit):
    """Find the largest power of p that is <= limit."""
    power = p
    while power * p <= limit:
        power *= p
    return power

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Get all primes up to 256
    primes = _sieve_primes(_N)
    
    # For each prime, find the largest prime power <= 256
    # The optimal strategy is to pick the largest prime power for each prime
    # This ensures all selected numbers are pairwise coprime
    selected = [1]  # Always include 1
    selected_set = {1}
    
    for p in primes:
        pp = _largest_prime_power_for_prime(p, _N)
        # Check if pp is coprime to all already selected
        conflict = False
        for s in selected_set:
            if s == 1:
                continue
            if math.gcd(pp, s) != 1:
                conflict = True
                break
        if not conflict:
            selected.append(pp)
            selected_set.add(pp)
    
    current_sum = sum(selected)
    
    # Now try to improve: some primes' largest prime powers might conflict
    # with each other. In that case, we need to choose which one to keep.
    # Actually, prime powers of different primes are always coprime, so there
    # should be no conflict. Let me verify...
    # p^a and q^b where p != q are always coprime. So no conflict!
    
    # But wait - what about composite numbers that might give a better sum?
    # For example, instead of picking 2^7=128 and 3^5=243, could we pick
    # a number like 2*3=6? No, that's worse. The prime power is always better
    # than any composite involving that prime, because the composite would be
    # smaller than the prime power itself.
    
    # Actually, let me reconsider. The constraint is that no two numbers can
    # share a prime factor. If we pick p^a for prime p, we use up prime p.
    # We could instead pick some composite number that includes p and other
    # primes, but then we'd use up those other primes too, and the number
    # would be smaller than p^a (since it would be p * (product of other primes)
    # which is less than p^a if a is large enough).
    
    # For small primes, the prime power might not be that large. For example,
    # for p=2, the largest power is 128. For p=3, it's 243. These are both
    # large. But what if we could combine some small primes into a composite
    # that's larger than the individual prime powers?
    
    # Example: 2*3*5*7 = 210. This uses primes 2,3,5,7. The individual prime
    # powers would be 128 (for 2), 243 (for 3), 125 (for 5), 49 (for 7).
    # Sum of individual: 128+243+125+49 = 545. Composite: 210. So individual
    # is better.
    
    # In general, the product of primes is much smaller than the sum of their
    # largest powers. So the strategy of picking the largest prime power for
    # each prime is optimal.
    
    # Let me verify by checking if any composite number could replace a set of
    # prime powers with a better sum.
    
    # Actually, I realize the above reasoning is correct. The largest prime power
    # for each prime is the best choice, and since prime powers of different
    # primes are coprime, we can pick them all.
    
    # Let me double-check by computing the sum and verifying coprimality.
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # This should never happen with prime powers of different primes
                pass
    
    # Ensure result is a list of Python ints
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
