"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Precompute small primes up to 1e6 for fast factorization
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(1000000)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes."""
        if x <= 0:
            return 0
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def check_run(start):
        """Check phi values for start to start+199, return length of distinct run."""
        seen = set()
        k = 0
        for i in range(200):
            val = phi_fast(start + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    def update_best(start, k):
        nonlocal best_k, best_s
        if k > best_k:
            best_k = k
            best_s = start
    
    # Strategy 1: Small range exhaustive scan (1 to 100000)
    if time.time() < deadline:
        scan_end = 100000
        # Compute phi for range 1..scan_end
        vals = [0] * (scan_end + 1)
        for x in range(1, scan_end + 1):
            vals[x] = phi_fast(x)
        
        # Find best run
        for s in range(1, scan_end - 199):
            seen = set()
            k = 0
            for i in range(s, min(s + 200, scan_end + 1)):
                if vals[i] in seen:
                    break
                seen.add(vals[i])
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
    
    # Strategy 2: Targeted search around "smooth" numbers and primorials
    if time.time() < deadline:
        candidates = []
        # Near primorials
        primorial = 2
        for p in primes[:15]:
            primorial *= p
            if primorial > 10**12:
                break
            for delta in [-10, -5, -1, 1, 5, 10, 20, 50]:
                s = primorial + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        # Near powers of 2
        for e in range(1, 40):
            p2 = 1 << e
            for delta in [-10, -5, -1, 1, 5, 10, 20, 50, 100]:
                s = p2 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        # Near powers of 3
        for e in range(1, 26):
            p3 = 3 ** e
            for delta in [-10, -5, -1, 1, 5, 10, 20, 50]:
                s = p3 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        # Deduplicate and sort
        candidates = sorted(set(candidates))
        
        for s in candidates:
            if time.time() >= deadline:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_run(s)
            update_best(s, k)
    
    # Strategy 3: Random sampling in large ranges - main search
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline - 10 and attempts < 100000:
            attempts += 1
            # Sample from various ranges with weighted distribution
            r = random.random()
            if r < 0.2:
                # Lower-mid range
                s = random.randint(10**4, 10**7)
            elif r < 0.4:
                # Mid-range
                s = random.randint(10**7, 10**9)
            elif r < 0.6:
                # Upper-mid range
                s = random.randint(10**9, 10**11)
            elif r < 0.8:
                # Upper range
                s = random.randint(10**11, 10**12 - 200)
            else:
                # Very high range
                s = random.randint(10**11, 10**12 - 200)
            
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_run(s)
            update_best(s, k)
            
            # If we found a long run, do local exploration
            if k > best_k - 2 and k > 10:
                # Explore nearby starts
                for delta in range(-5, 6):
                    if time.time() >= deadline - 5:
                        break
                    ns = s + delta
                    if 1 <= ns <= 10**12 - 200:
                        nk = check_run(ns)
                        update_best(ns, nk)
    
    # Strategy 4: Local search around current best
    if time.time() < deadline - 5 and best_k > 5:
        center = best_s
        for delta in range(-100, 101, 2):
            if time.time() >= deadline - 3:
                break
            s = center + delta
            if 1 <= s <= 10**12 - 200:
                k = check_run(s)
                update_best(s, k)
    
    # Strategy 5: If we have time, do a denser scan in promising regions
    if time.time() < deadline - 15:
        # Focus on region around current best
        if best_k >= 3:
            lo = max(1, best_s - 2000)
            hi = min(10**12 - 200, best_s + 2000)
            # Scan in chunks of 500
            s = lo
            while s + 500 <= hi + 1 and time.time() < deadline - 5:
                for i in range(s, min(s + 500, hi + 1)):
                    if time.time() >= deadline - 3:
                        break
                    k = check_run(i)
                    update_best(i, k)
                s += 500
    
    # Ensure validity
    s = best_s
    k = best_k
    if s < 1:
        s = 1
    if s + k - 1 > 10**12:
        k = 10**12 - s + 1
        if k < 1:
            s = 1
            k = 1
    if k > 200:
        k = 200
    if k < 1:
        k = 1
    
    return {'start': s, 'length': k}

# EVOLVE-BLOCK-END
