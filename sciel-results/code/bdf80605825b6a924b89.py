"""Rewritten program: segmented sieve for p-smoothness checking, broad prime search."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Deterministic witnesses for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # ---------- Sieve primes up to limit ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to 10^6 for trial division and segmented sieve
    small_primes = sieve_primes(1000000)

    # ---------- Segmented sieve: factor numbers in [p^2, p^2 + max_k] ----------
    def check_p_smooth(p, max_k=200):
        """Check p^2+1, ..., p^2+max_k for p-smoothness using segmented sieve."""
        lo = p * p
        hi = lo + max_k
        size = hi - lo + 1
        
        # Initialize: store the number itself if not divisible by small primes
        # We'll track the remaining factor for each position
        rem = list(range(lo + 1, hi + 1))  # rem[i] = remaining factor of lo+1+i
        
        # Divide out all small primes <= p
        for sp in small_primes:
            if sp > p:
                break
            # Find first multiple of sp in [lo+1, hi]
            first = ((lo + 1 + sp - 1) // sp) * sp
            if first > hi:
                continue
            for j in range(first - lo - 1, size - 1, sp):
                while rem[j] % sp == 0:
                    rem[j] //= sp
                    # If rem[j] becomes 1, it's fully factored
                    if rem[j] == 1:
                        break
        
        # Now check each remaining value
        k = 0
        for i in range(1, max_k + 1):  # i corresponds to p^2 + i
            idx = i  # rem[i] is the remainder for lo + 1 + i = p^2 + i... wait
            # Actually rem[0] corresponds to lo+1 = p^2+1
            # rem[j] where j = (p^2 + i) - (lo+1) = i - 1
            val = rem[i - 1]
            if val == 1:
                k += 1
            else:
                # val is the remaining factor; check if it's <= p
                # If val is prime and > p, fail
                # If val is composite, it has a factor > 1 that we couldn't divide out
                # Since we divided all primes <= p, val must be either 1 or a prime > p
                # (or product of primes > p, but then smallest factor > p)
                if val <= p:
                    k += 1
                else:
                    break
        
        return k

    # ---------- Fast p-smoothness check for a single number ----------
    def is_p_smooth(n, p, small_primes):
        """Check if all prime factors of n are <= p."""
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > n:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        # n is either prime or product of primes > all small primes we tried
        # If n <= p, it's fine (it's a prime <= p)
        if n <= p:
            return True
        # n > p: check if n is prime (then it's a prime factor > p)
        # If n is composite, it has factors, but since we divided all primes <= p,
        # any factor of n must be > p, so n is not p-smooth
        return False

    # ---------- Main search ----------
    best_k = 0
    best_p = 2

    # Phase 1: Search small primes using segmented sieve (fast)
    # Primes up to ~10^6 can be checked very quickly
    for p in small_primes:
        if time.time() > deadline:
            break
        if p > 1000000:
            break
        k = check_p_smooth(p, 200)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes using probabilistic approach
    # For large p, p^2 + i for small i is very likely to have small factors
    # Strategy: for each candidate p, check p^2+1, p^2+2, ... sequentially
    # Use trial division by small primes first, then Miller-Rabin for remainder
    
    # Try primes in ranges, skipping ahead
    # For p > 10^6, we sample primes
    search_limit = 10**9
    step = 10000  # check every 10000th odd number as potential prime
    
    # Generate candidate primes in larger ranges
    # Focus on ranges where we might find good k
    # Heuristic: try primes up to 10^7 first, then sample beyond
    
    # Continue from where phase 1 left off
    next_p = 1000003  # next prime after 10^6
    
    # For efficiency, use a simple prime test and check
    while next_p < search_limit:
        if time.time() > deadline:
            break
        
        # Check a batch of candidate primes
        for offset in range(0, 200000, 2):
            if time.time() > deadline:
                break
            candidate = next_p + offset
            if candidate > search_limit:
                break
            
            # Quick primality test
            if not is_prime_mr(candidate):
                continue
            
            p = candidate
            # Check p-smoothness for p^2+1, ..., p^2+200
            k = 0
            for i in range(1, 201):
                n = p * p + i
                # Trial divide by small primes
                temp = n
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > temp:
                        break
                    while temp % sp == 0:
                        temp //= sp
                if temp == 1:
                    k += 1
                elif temp <= p:
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
        
        next_p += 200000

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
