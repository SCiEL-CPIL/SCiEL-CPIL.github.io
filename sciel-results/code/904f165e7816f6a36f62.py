"""Rewritten program using segmented sieve for SPF over [p^2+1, p^2+200]."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Strategy: Use segmented sieve to compute SPF for [p^2+1, p^2+200]
    # Then verify p-smoothness by checking if all prime factors <= p.
    
    # Key insight: For n in [p^2+1, p^2+200], if n has a prime factor > p,
    # that factor must be in (p, n]. Since n < p^2 + p for p > 200,
    # any prime factor > p must satisfy p < q <= p^2+200.
    # 
    # Better approach: Precompute primes up to sqrt(max_n) where max_n = p^2+200.
    # For each candidate p, use segmented sieve to factor the 200 numbers.
    
    def sieve_primes(limit):
        """Return list of primes up to limit using bytearray sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start::step] = bytearray(len(is_prime[start::step]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We'll try primes up to a reasonable limit.
    # For each prime p, we need to check if p^2+1, ..., p^2+k are all p-smooth.
    # 
    # Efficient check: For a given p, compute the largest prime factor of each n in [p^2+1, p^2+200].
    # If LPF(n) <= p for all n in the range, then they're all p-smooth.
    #
    # To compute LPF efficiently for 200 consecutive numbers near p^2:
    # 1. Precompute all primes up to some limit (say 10^6) for trial division.
    # 2. For each candidate p, for each n in [p^2+1, p^2+200]:
    #    - Divide n by all primes <= min(p, sqrt(n)) 
    #    - If remaining factor > 1 and > p, not p-smooth
    #
    # But this is still O(pi(p)) per number. Let's use a smarter approach.
    #
    # Smarter: Use the fact that for n near p^2, any prime factor q > p must have
    # n/q < p. So if we can show that no prime q in (p, n] divides n, we're good.
    #
    # Even better: Precompute primes up to a large limit. For each candidate p,
    # use a segmented sieve over [p^2+1, p^2+200] using primes up to p.
    # After sieving, if any number > 1 remains, check if it's prime and <= p.
    
    max_search_p = 500000
    primes = sieve_primes(max_search_p)
    
    best_k = 0
    best_p = 0
    
    # For each prime p, check the longest run of p-smooth numbers
    for p in primes:
        p2 = p * p
        
        # Create array of 200 values: v[i] = p^2 + 1 + i for i=0..199
        # We'll divide these by all primes <= p
        # Use a list for the 200 values
        vals = [p2 + 1 + i for i in range(200)]
        
        # For each prime q <= p, divide vals[i] by q if divisible
        # Optimization: only iterate primes up to p
        for q in primes:
            if q > p:
                break
            # For each i, if vals[i] % q == 0, divide out all factors of q
            for i in range(200):
                if vals[i] % q == 0:
                    while vals[i] % q == 0:
                        vals[i] //= q
        
        # Now check how many leading values are 1 (fully factored)
        k = 0
        for i in range(200):
            if vals[i] == 1:
                k += 1
            else:
                # vals[i] > 1 means there's a prime factor > p
                # (since we divided out all primes <= p)
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    if best_k >= 1:
        return {'p': best_p, 'k': best_k}
    
    return {'p': 3, 'k': 1}

# EVOLVE-BLOCK-END
