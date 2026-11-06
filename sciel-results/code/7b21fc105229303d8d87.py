"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1).
    
    Strategy:
    1. Generate small primes up to a threshold P.
    2. Use a segmented sieve to identify numbers in [10000, 1000000] that are P-smooth
       (all prime factors <= P).
    3. Find consecutive pairs (n, n+1) where both are P-smooth.
    4. Among these, compute the exact largest prime factor of n*(n+1) via trial division
       and return the n with the minimum such value.
    """
    LO = 10000
    HI = 1000000
    
    # Step 1: Generate small primes up to threshold P
    # We want P large enough to catch smooth numbers but small enough for efficiency
    P = 200
    
    # Sieve of Eratosthenes for primes up to P
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    small_primes = sieve_primes(P)
    
    # Step 2: Segmented sieve to find P-smooth numbers in [LO, HI]
    # A number is P-smooth if it has no prime factor > P
    # We mark numbers that are NOT P-smooth by checking divisibility by primes > P
    
    # For efficiency, we'll work in segments
    segment_size = 100000
    smooth_flags = {}  # n -> True if n is P-smooth
    
    # We need to check if a number has any prime factor > P
    # Equivalent: after dividing out all prime factors <= P, if remainder is 1, it's P-smooth
    
    def is_p_smooth(n, primes):
        """Check if n is P-smooth by trial division with primes up to P."""
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                temp //= p
        return temp == 1
    
    # Process in segments to manage memory
    candidates = []  # (n, lpf_of_n, lpf_of_n_plus_1)
    
    for seg_start in range(LO, HI + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, HI)
        
        # For each n in segment, check if n and n+1 are P-smooth
        for n in range(seg_start, seg_end + 1):
            # Quick check: if n is not P-smooth, skip
            if not is_p_smooth(n, small_primes):
                continue
            if not is_p_smooth(n + 1, small_primes):
                continue
            
            # Both are P-smooth; compute exact largest prime factor
            def largest_prime_factor(x):
                if x <= 1:
                    return 1
                lpf = 1
                d = 2
                while d * d <= x:
                    if x % d == 0:
                        lpf = d
                        while x % d == 0:
                            x //= d
                    d += 1 if d == 2 else 2  # skip even numbers after 2
                if x > 1:
                    lpf = x
                return lpf
            
            lpf_n = largest_prime_factor(n)
            lpf_n1 = largest_prime_factor(n + 1)
            lpf_product = max(lpf_n, lpf_n1)
            candidates.append((lpf_product, n))
    
    if not candidates:
        # Fallback: return a reasonable default
        return 10000
    
    # Return n with minimum largest prime factor
    candidates.sort()
    return candidates[0][1]
# EVOLVE-BLOCK-END
