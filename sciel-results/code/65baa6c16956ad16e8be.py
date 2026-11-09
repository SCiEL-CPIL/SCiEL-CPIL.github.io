"""Rewritten program: search for p with long p-smooth run after p^2."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # --- Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981 ---
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime_mr(n):
        if n < 2:
            return False
        for p in _MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
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

    # --- Pollard's Rho / Brent variant ---
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
            m = 128
            g = r = q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    # --- Full factorization ---
    def factorize(n):
        """Return list of prime factors (with multiplicity) of n."""
        factors = []
        # small trial division
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        # recursive decomposition
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors

    # --- Check largest prime factor of v ---
    def largest_prime_factor(v):
        if v <= 1:
            return 1
        # quick small-prime trial division
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while v % p == 0:
                v //= p
        if v == 1:
            return 97
        if is_prime_mr(v):
            return v
        # factorize remaining
        facs = factorize(v)
        return max(facs)

    def max_factor_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                n //= p
        if n == 1:
            return True
        if n > limit:
            # n might be prime > limit, or composite with a factor > limit
            # check if n itself is prime
            if is_prime_mr(n):
                return n <= limit
            # factorize
            facs = factorize(n)
            return all(f <= limit for f in facs)
        else:
            # n <= limit, so all factors are <= n <= limit
            return True

    def run_length(p, max_k=200):
        """Find the longest run k such that p^2+1, ..., p^2+k all have all prime factors <= p."""
        k = 0
        base = p * p
        for j in range(1, max_k + 1):
            if not max_factor_leq(base + j, p):
                break
            k = j
        return k

    # --- Segmented sieve to generate primes in a range ---
    def segmented_primes(lo, hi, small_primes_list):
        """Yield primes in [lo, hi] using segmented sieve with small_primes_list as base primes."""
        size = hi - lo + 1
        if size <= 0:
            return
        is_p = bytearray(b'\x01') * size
        if lo == 1:
            is_p[0] = 0
        # mark composites
        for p in small_primes_list:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - lo, size, p):
                is_p[j] = 0
        for i in range(size):
            if is_p[i]:
                yield lo + i

    # --- Main search ---
    # Precompute small primes up to 10^6 for segmented sieve
    SMALL_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SMALL_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SMALL_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(SMALL_LIMIT + 1) if sieve[i]]

    best_k = 0
    best_p = 2

    # Strategy: search primes in increasing segments, checking run lengths.
    # Larger primes make the condition easier (p-bound is larger).
    # We'll search from small to large, but prioritize larger primes since they're more likely to have long runs.
    # Actually, let's search from larger primes first to find good candidates quickly,
    # then verify and potentially find even better ones.

    # Let's do a two-phase approach:
    # Phase 1: Search a range of larger primes (e.g., 10^6 to 10^7) in segments
    # Phase 2: If time remains, search other ranges

    # Search segments from high to low
    seg_size = 500000
    search_lo = 10**6
    search_hi = 10**7

    # Iterate segments from high end
    for seg_start in range(search_hi - seg_size, search_lo - 1, -seg_size):
        if time.time() > deadline:
            break
        seg_lo = max(seg_start, search_lo)
        seg_hi = min(seg_start + seg_size, search_hi)
        if seg_lo >= seg_hi:
            continue

        for p in segmented_primes(seg_lo, seg_hi, small_primes):
            if time.time() > deadline:
                break
            k = run_length(p, 200)
            if k > best_k:
                best_k = k
                best_p = p

    # If we have time, also search smaller primes
    if time.time() < deadline - 5:
        # Search primes from 10^5 to 10^6
        for seg_start in range(10**6 - seg_size, 10**5 - 1, -seg_size):
            if time.time() > deadline:
                break
            seg_lo = max(seg_start, 10**5)
            seg_hi = min(seg_start + seg_size, 10**6)
            if seg_lo >= seg_hi:
                continue
            for p in segmented_primes(seg_lo, seg_hi, small_primes):
                if time.time() > deadline:
                    break
                k = run_length(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p

    # Also check very small primes
    if time.time() < deadline - 2:
        for p in small_primes:
            if p > 10**5:
                break
            if time.time() > deadline:
                break
            k = run_length(p, 200)
            if k > best_k:
                best_k = k
                best_p = p

    if best_k < 1:
        # Fallback: at least p=2 gives p^2+1=3, which has prime factor 3 > 2. So k=0.
        # But we need k >= 1. Let's check p=3: p^2+1=10=2*5, 5>3. k=0.
        # p=5: 26=2*13, 13>5. k=0.
        # p=7: 50=2*5^2, all <=7. k=1. 51=3*17, 17>7. So k=1.
        # p=11: 122=2*61, 61>11. k=0.
        # p=13: 170=2*5*17, 17>13. k=0.
        # p=17: 290=2*5*29, 29>17. k=0.
        # p=19: 362=2*181, 181>19. k=0.
        # p=23: 530=2*5*53, 53>23. k=0.
        # p=29: 842=2*421, 421>29. k=0.
        # p=31: 962=2*13*37, 37>31. k=0.
        # p=37: 1370=2*5*137, 137>37. k=0.
        # p=41: 1682=2*283, 283>41. k=0.
        # p=43: 1850=2*5^2*37, all <=43. k=1. 1851=3*617, 617>43. k=1.
        # p=47: 2210=2*5*13*17, all <=47. k=1. 2211=3*737=3*11*67, 67>47. k=1.
        # p=53: 2810=2*5*281, 281>53. k=0.
        # p=59: 3482=2*13^2*11... wait 3482/2=1741. 1741/13=134? no. 1741 is prime? 1741/7=248.7, /11=158.3, /13=133.9, /17=102.4, /19=91.6, /23=75.7, /29=60.03, /31=56.16, /37=47.05, /41=42.46, /43=40.49, /47=37.04. sqrt(1741)~41.7. So 1741 is prime. 1741>59. k=0.
        # Let's just set a known good one
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
