"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill
    
    def now():
        return time.time()
    
    # Fast primality test using Miller-Rabin for large numbers
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses good for 32-bit; for 64-bit we need more
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's Rho for factoring
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            # retry with different x, c
    
    import random
    
    # Factorize n into prime factors (with multiplicity)
    def factorize(n):
        if n == 1:
            return []
        factors = []
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        # Use Pollard's rho
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_rho(m)
                stack.append(d)
                stack.append(m // d)
        return factors
    
    # Compute phi(n) using factorization
    def phi(n):
        if n == 1:
            return 1
        factors = factorize(n)
        result = n
        seen_primes = set()
        for p in factors:
            if p not in seen_primes:
                result -= result // p
                seen_primes.add(p)
        return result
    
    # Batch phi computation for a range [lo, hi] using segmented approach
    def phi_batch(lo, hi):
        """Compute phi(n) for all n in [lo, hi]."""
        size = hi - lo + 1
        if size <= 0:
            return []
        
        # For each number in the range, compute phi
        phi_vals = []
        for n in range(lo, hi + 1):
            phi_vals.append(phi(n))
        return phi_vals
    
    # Search for best window
    best_start = 1
    best_len = 1
    
    def update_best(s, k):
        nonlocal best_start, best_len
        if k > best_len:
            best_len = k
            best_start = s
    
    # Phase 1: Quick search in small ranges to establish baseline
    # Use the original approach but faster
    def search_small_range(end):
        nonlocal best_start, best_len
        values = [0] + [phi(x) for x in range(1, end + 1)]
        for s in range(1, end - 1):
            seen, k = set(), 0
            while s + k <= end and values[s + k] not in seen:
                seen.add(values[s + k])
                k += 1
            update_best(s, k)
    
    # Phase 2: Search in larger ranges using sliding window
    def search_range(lo, hi, max_window=200):
        nonlocal best_start, best_len
        
        # We'll process in chunks
        chunk_size = max(100, max_window * 2)
        
        pos = lo
        while pos <= hi and now() < DEADLINE:
            chunk_end = min(hi, pos + chunk_size - 1)
            if chunk_end - pos + 1 < 1:
                break
            
            # Compute phi for this chunk
            vals = phi_batch(pos, chunk_end)
            n_vals = len(vals)
            
            # Slide a window of size up to max_window
            for i in range(n_vals):
                if now() > DEADLINE:
                    return
                # Check if extending from i gives a better result
                max_check = min(max_window, n_vals - i)
                
                # Early termination: if even with full window we can't beat best
                if max_check <= best_len:
                    continue
                
                seen = set()
                k = 0
                for j in range(i, i + max_check):
                    v = vals[j]
                    if v in seen:
                        break
                    seen.add(v)
                    k += 1
                
                if k > best_len:
                    best_len = k
                    best_start = pos + i
            
            # Move to next position
            pos = chunk_end + 1 - max_window + 1  # overlap to catch windows crossing chunk boundaries
    
    # Execute search phases
    # Phase 1: Small ranges
    for end in [100, 500, 2000, 10000, 50000]:
        if now() > DEADLINE:
            break
        search_small_range(end)
    
    # Phase 2: Larger ranges - search in strategic locations
    # For larger n, phi values are larger and collisions less likely
    # Search in various ranges up to 10^12
    
    # Try ranges around powers and other strategic points
    search_regions = []
    # Small ranges already done
    # Medium ranges
    search_regions.append((100000, 200000))
    search_regions.append((500000, 600000))
    search_regions.append((1000000, 1100000))
    # Larger ranges
    search_regions.append((10000000, 10100000))
    search_regions.append((100000000, 100100000))
    search_regions.append((1000000000, 1000100000))
    # Very large ranges
    search_regions.append((10000000000, 10000100000))
    search_regions.append((100000000000, 100000100000))
    search_regions.append((1000000000000, 1000000100000))
    
    for lo, hi in search_regions:
        if now() > DEADLINE:
            break
        if hi > 10**12:
            hi = 10**12
        if lo >= hi:
            continue
        search_range(lo, hi, max_window=200)
    
    # Final check
    if best_len < 1:
        best_len = 1
        best_start = 1
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
