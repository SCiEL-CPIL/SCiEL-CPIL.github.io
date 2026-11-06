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
    
    def compute_phi_range(start, length):
        """Compute phi values for range [start, start+length-1]."""
        return [phi_fast(start + i) for i in range(length)]
    
    def find_best_run(values, max_len=200):
        """Find the longest run of consecutive distinct phi values in values."""
        n = len(values)
        best_k = 1
        best_s = 0
        # Sliding window approach
        seen = {}
        left = 0
        for right in range(n):
            val = values[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            if best_k >= max_len:
                break
        return best_k, best_s

    best_overall = (1, 1)  # (length, start)
    
    def check_and_update(start, length):
        nonlocal best_overall
        if length < best_overall[0]:
            return
        vals = compute_phi_range(start, length)
        k, offset = find_best_run(vals, length)
        if k > best_overall[0]:
            best_overall = (k, start + offset)

    # Strategy 1: Dense scan of small range using optimized phi
    # We scan up to 10^6 first, then extend if time permits.
    # Using a sieve for phi is faster for dense ranges.
    
    def compute_phi_sieve(limit):
        phi = list(range(limit + 1))
        for p in primes:
            if p * p > limit:
                break
            for j in range(p, limit + 1, p):
                phi[j] -= phi[j] // p
        # Handle remaining primes? 
        # The standard sieve for phi works if we iterate through all primes up to limit.
        # My primes list only goes to 1e6.
        # If limit <= 1e6, this is correct.
        # If limit > 1e6, we need primes up to limit.
        # For limit=1e6, it's fine.
        return phi

    scan_limit = 1000000
    if time.time() < deadline:
        phi_vals = compute_phi_sieve(scan_limit)
        # phi_vals[0] is 0, phi_vals[1] is 1.
        # We want to find the best run in phi_vals[1:]
        # But find_best_run expects a list of values.
        # We can pass phi_vals[1:]
        k, offset = find_best_run(phi_vals[1:], 200)
        if k > best_overall[0]:
            best_overall = (k, offset + 1)
        
        # Try to extend scan if time permits
        # But computing phi for 10^7 with Python loop is slow.
        # Let's stick to 10^6 for now and use random for larger.
    
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
            check_and_update(s, 200)
    
    # Strategy 3: Random sampling in large ranges
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 50000:
            attempts += 1
            # Sample from various ranges
            r = random.random()
            if r < 0.3:
                # Mid-range
                s = random.randint(10**6, 10**9)
            elif r < 0.6:
                # Upper range
                s = random.randint(10**9, 10**12 - 200)
            elif r < 0.8:
                # Lower-mid range
                s = random.randint(10**4, 10**7)
            else:
                # Very high range
                s = random.randint(10**11, 10**12 - 200)
            
            if s < 1 or s + 200 > 10**12:
                continue
            check_and_update(s, 200)
    
    # Strategy 4: Local search - if we found a good run, explore nearby
    if time.time() < deadline and best_overall[0] > 5:
        center = best_overall[1]
        for delta in range(-200, 201, 10):
            if time.time() >= deadline:
                break
            s = center + delta
            if 1 <= s <= 10**12 - 200:
                check_and_update(s, 200)
    
    # Strategy 5: If we have time, do a denser scan in promising regions
    if time.time() < deadline - 10:
        # Focus on region around current best
        if best_overall[0] >= 3:
            lo = max(1, best_overall[1] - 5000)
            hi = min(10**12 - 200, best_overall[1] + 5000)
            # Scan in chunks
            chunk = 1000
            s = lo
            while s + chunk <= hi + 1 and time.time() < deadline - 5:
                vals = compute_phi_range(s, chunk)
                k, offset = find_best_run(vals, 200)
                if k > best_overall[0]:
                    best_overall = (k, s + offset)
                s += chunk
    
    # Ensure validity
    s = best_overall[1]
    k = best_overall[0]
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
