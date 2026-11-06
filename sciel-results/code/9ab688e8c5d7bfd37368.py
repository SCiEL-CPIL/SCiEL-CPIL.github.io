import time
import math
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Precompute small primes for trial division ---
    LIMIT_TRIAL = 10**6
    is_prime_small = bytearray(b'\x01') * (LIMIT_TRIAL + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(math.isqrt(LIMIT_TRIAL)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = bytearray(len(is_prime_small[i*i::i]))
    small_primes = [i for i in range(2, LIMIT_TRIAL + 1) if is_prime_small[i]]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < LIMIT_TRIAL:
            return bool(is_prime_small[n])
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

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

    def factorize(n):
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi_large(n):
        if n == 1:
            return 1
        temp = n
        result = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                factors = factorize(temp)
                for p in set(factors):
                    result -= result // p
        return result

    def phi_batch(start, count):
        """Compute phi for count consecutive integers starting at start."""
        return [phi_large(start + i) for i in range(count)]

    def max_distinct_run(vals, max_k=200):
        """Find the longest run of consecutive distinct values in vals."""
        n = len(vals)
        best_k = 0
        best_idx = 0
        
        for i in range(n):
            if n - i < best_k:
                break
            seen = set()
            for j in range(i, min(i + max_k, n)):
                v = vals[j]
                if v in seen:
                    break
                seen.add(v)
                curr_k = j - i + 1
                if curr_k > best_k:
                    best_k = curr_k
                    best_idx = i
                    if best_k == max_k:
                        return best_k, best_idx
        return best_k, best_idx

    def sliding_window_max_distinct(phi_arr, max_k=200):
        """Use sliding window with Counter to find max distinct run efficiently."""
        n = len(phi_arr)
        if n == 0:
            return 0, 0
        
        best_k = 0
        best_idx = 0
        counts = Counter()
        distinct = 0
        
        # Initialize window [0, max_k)
        for i in range(min(max_k, n)):
            v = phi_arr[i]
            counts[v] += 1
            if counts[v] == 1:
                distinct += 1
        
        if distinct > best_k:
            best_k = distinct
            best_idx = 0
        
        # Slide window
        for left in range(1, n - max_k + 1):
            # Remove left-1
            v_out = phi_arr[left - 1]
            counts[v_out] -= 1
            if counts[v_out] == 0:
                del counts[v_out]
                distinct -= 1
            
            # Add left + max_k - 1
            v_in = phi_arr[left + max_k - 1]
            counts[v_in] += 1
            if counts[v_in] == 1:
                distinct += 1
            
            if distinct > best_k:
                best_k = distinct
                best_idx = left
                if best_k == max_k:
                    break
        
        # Also check shorter windows at the end if needed
        # But for our purposes, the sliding window of size max_k is sufficient
        # if we found a full window. For shorter runs, we need to check.
        
        # Check for shorter runs by scanning
        # Actually, let's do a proper scan for the best run of any length up to max_k
        # But the sliding window above only checks windows of exactly max_k size.
        # We need to find the longest run where all values are distinct.
        
        # Let's redo with a proper two-pointer approach
        best_k = 0
        best_idx = 0
        counts = {}
        left = 0
        
        for right in range(n):
            v = phi_arr[right]
            counts[v] = counts.get(v, 0) + 1
            
            # Shrink window until all distinct
            while counts[v] > 1:
                v_left = phi_arr[left]
                counts[v_left] -= 1
                if counts[v_left] == 0:
                    del counts[v_left]
                left += 1
            
            curr_k = right - left + 1
            if curr_k > best_k:
                best_k = curr_k
                best_idx = left
                if best_k >= max_k:
                    break
        
        return best_k, best_idx

    best_k = 1
    best_s = 1

    # --- Phase 1: Fast NumPy Sieve with Sliding Window ---
    # Compute phi for a large range using vectorized operations
    
    N = 20 * 10**6  # 20 million
    
    try:
        # Initialize phi array
        phi_arr = np.arange(N + 1, dtype=np.int64)
        
        # Sieve: for each prime p, update phi for all multiples of p
        # Use a bytearray to track which indices are prime
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = is_prime[1] = 0
        
        # Find primes using standard sieve
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        
        # Now update phi for each prime
        primes = [i for i in range(2, N + 1) if is_prime[i]]
        
        for p in primes:
            # Update phi for all multiples of p
            # phi[m] -= phi[m] // p for all m that are multiples of p
            multiples = np.arange(p, N + 1, p)
            phi_arr[multiples] -= phi_arr[multiples] // p
        
        if now() > DEADLINE - 30.0:
            # If we're running out of time, use what we have
            pass
        
        # Find max distinct run using sliding window
        vals = phi_arr[1:]
        k, idx = sliding_window_max_distinct(vals, 200)
        if k > best_k:
            best_k = k
            best_s = idx + 1
            
    except Exception as e:
        # Fallback to smaller N
        N = 5 * 10**6
        try:
            phi_arr = np.arange(N + 1, dtype=np.int64)
            is_prime = bytearray(b'\x01') * (N + 1)
            is_prime[0] = is_prime[1] = 0
            for i in range(2, int(math.isqrt(N)) + 1):
                if is_prime[i]:
                    is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
            primes = [i for i in range(2, N + 1) if is_prime[i]]
            for p in primes:
                multiples = np.arange(p, N + 1, p)
                phi_arr[multiples] -= phi_arr[multiples] // p
            vals = phi_arr[1:]
            k, idx = sliding_window_max_distinct(vals, 200)
            if k > best_k:
                best_k = k
                best_s = idx + 1
        except:
            pass

    # --- Phase 2: Targeted Search in Larger Ranges ---
    if now() < DEADLINE - 10.0:
        random.seed(42)
        
        search_regions = []
        
        # Powers of 10
        for exp in range(6, 13):
            base = 10 ** exp
            for offset in [-1000, -500, -100, 0, 100, 500, 1000]:
                s = base + offset
                if 1 <= s <= 10**12 - 200:
                    search_regions.append(s)
        
        # Random large ranges
        for _ in range(50):
            s = random.randint(10**6, 10**12 - 200)
            search_regions.append(s)
        
        # Also check some medium ranges
        for _ in range(20):
            s = random.randint(10**5, 10**7)
            search_regions.append(s)
        
        search_regions.sort(reverse=True)
        
        for s in search_regions:
            if now() > DEADLINE - 5.0:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            
            try:
                vals = phi_batch(s, 200)
            except Exception:
                continue
            
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = s + idx
                if best_k == 200:
                    break
    
    # --- Phase 3: Fine-tune around best found ---
    if now() < DEADLINE - 3.0 and best_k < 200:
        window = 5000
        s_min = max(1, best_s - window)
        s_max = min(10**12 - 200, best_s + window)
        
        n_samples = 50
        for i in range(n_samples):
            if now() > DEADLINE - 2.0:
                break
            s = s_min + (i * (s_max - s_min)) // max(1, n_samples - 1)
            s = int(s)
            if s < 1 or s + 200 > 10**12:
                continue
            try:
                vals = phi_batch(s, 200)
            except Exception:
                continue
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = s + idx
                if best_k == 200:
                    break

    return {'start': best_s, 'length': best_k}
