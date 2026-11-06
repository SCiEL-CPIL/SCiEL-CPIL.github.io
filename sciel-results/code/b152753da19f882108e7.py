import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in small_primes:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
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
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    def factorize(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    # Sieve primes up to a limit
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Check if n has all prime factors <= p
    def all_factors_leq(n, p, small_primes_list):
        if n <= 1:
            return True
        if n <= p:
            return True
        # Trial divide by small primes up to min(p, some_limit)
        limit = min(p, 10000)
        for sp in small_primes_list:
            if sp > limit:
                break
            if sp > n:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        # Now n has no prime factors <= limit
        # Check if n is prime
        if is_prime(n):
            return n <= p
        # n is composite, has a prime factor > limit
        # Factor n and check
        factors = factorize(n)
        return max(factors) <= p

    # Precompute small primes for trial division
    small_primes_list = sieve(10000)

    best_k = 0
    best_p = 2

    # First, do a quick search over primes to establish a baseline
    small_primes_to_check = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes_to_check:
        k = 0
        while k < 200:
            n = p * p + k + 1
            if all_factors_leq(n, p, small_primes_list):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Now use logarithmic lattice approach
    # Generate p-smooth numbers using log-space enumeration
    # For each prime p, we want to find runs of p-smooth numbers starting at p^2+1
    
    # Key insight: if N is p-smooth and N-1 = p^2, then N = p^2+1
    # We can generate smooth numbers and check if N-1 is a perfect square of a prime
    
    # Use a priority queue approach to generate smooth numbers in order
    # But we need to be smart about which p to check
    
    # Alternative: for each candidate p, generate the sequence p^2+1, p^2+2, ... 
    # and check smoothness, but do it more efficiently by precomputing
    
    # Let's try a different approach: enumerate potential p values where p^2+1 might be p-smooth
    # p^2+1 being p-smooth means all prime factors of p^2+1 are <= p
    
    # For larger p, p^2+1 is roughly p^2, so it can have at most 2 prime factors <= p
    # (since p^2 / (p * p) = 1)
    # Actually p^2+1 could be prime, or product of two primes both <= p
    
    # Let's search primes more systematically with better time management
    limits = [5*10**4, 2*10**5, 5*10**5, 10**6, 3*10**6, 5*10**6, 10**7]
    prev_limit = 100
    for limit in limits:
        if time.time() - start_time > deadline - 10:
            break
        
        primes = sieve(limit)
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            # Quick check: if p^2+1 has a large prime factor, k=0
            n = p * p + 1
            if not all_factors_leq(n, p, small_primes_list):
                continue
            
            k = 1
            while k < 200:
                n = p * p + k + 1
                if all_factors_leq(n, p, small_primes_list):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                # Update deadline based on progress
                elapsed = time.time() - start_time
                if elapsed > deadline - 20:
                    break
        
        prev_limit = limit
        if time.time() - start_time > deadline - 20:
            break

    # Additional targeted search: look for primes where p^2+1 factors nicely
    # Try some specific ranges where smooth numbers might cluster
    if time.time() - start_time < deadline - 30:
        # Search in ranges around powers of 2 and other structured values
        target_ranges = [(10**5, 2*10**5), (2*10**5, 5*10**5), (5*10**5, 10**6), 
                         (10**6, 2*10**6), (2*10**6, 5*10**6), (5*10**6, 10**7)]
        for lo, hi in target_ranges:
            if time.time() - start_time > deadline - 10:
                break
            primes = sieve(hi)
            for p in primes:
                if p < lo or p <= prev_limit:
                    continue
                if time.time() - start_time > deadline - 5:
                    break
                
                n = p * p + 1
                if not all_factors_leq(n, p, small_primes_list):
                    continue
                
                k = 1
                while k < 200:
                    n = p * p + k + 1
                    if all_factors_leq(n, p, small_primes_list):
                        k += 1
                    else:
                        break
                
                if k > best_k:
                    best_k = k
                    best_p = p
                    if time.time() - start_time > deadline - 15:
                        break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
