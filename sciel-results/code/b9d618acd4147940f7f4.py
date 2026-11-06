import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Precompute small primes for trial division ---
    LIMIT_TRIAL = 10**5
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

    best_k = 1
    best_s = 1

    # --- Phase 1: Optimized NumPy Sieve ---
    # Use a larger range. The previous loop was O(N). Now we iterate only over primes.
    # We can afford N = 10^7 or more.
    
    N = 10 * 10**6  # 10 million
    
    try:
        # 1. Generate primes up to N using a fast sieve
        is_prime_arr = bytearray(b'\x01') * (N + 1)
        is_prime_arr[0] = is_prime_arr[1] = 0
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime_arr[i]:
                is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
        
        # Get list of primes
        primes = [i for i in range(2, N + 1) if is_prime_arr[i]]
        
        # 2. Compute phi using only primes
        phi_arr = np.arange(N + 1, dtype=np.int64)
        
        for p in primes:
            # Update all multiples of p
            phi_arr[p::p] -= phi_arr[p::p] // p
            
            # Check time periodically in the loop? 
            # The loop over primes is ~664k iterations for 10^7.
            # Each iteration is a numpy slice op.
            # This should be fast.
            
        # Find max distinct run
        vals = phi_arr[1:]
        n_vals = len(vals)
        
        k, idx = max_distinct_run(vals, 200)
        if k > best_k:
            best_k = k
            best_s = idx + 1
            
    except Exception as e:
        # Fallback to smaller N if memory or time issue
        N = 5 * 10**6
        try:
            is_prime_arr = bytearray(b'\x01') * (N + 1)
            is_prime_arr[0] = is_prime_arr[1] = 0
            for i in range(2, int(math.isqrt(N)) + 1):
                if is_prime_arr[i]:
                    is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
            primes = [i for i in range(2, N + 1) if is_prime_arr[i]]
            
            phi_arr = np.arange(N + 1, dtype=np.int64)
            for p in primes:
                phi_arr[p::p] -= phi_arr[p::p] // p
                
            vals = phi_arr[1:]
            k, idx = max_distinct_run(vals, 200)
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
        # Increase count if time allows
        num_random_large = 100
        for _ in range(num_random_large):
            s = random.randint(10**6, 10**12 - 200)
            search_regions.append(s)
        
        # Medium ranges
        num_random_med = 50
        for _ in range(num_random_med):
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
        window = 1000
        s_min = max(1, best_s - window)
        s_max = min(10**12 - 200, best_s + window)
        
        for _ in range(50): # Increased from 20
            if now() > DEADLINE - 2.0:
                break
            s = random.randint(s_min, s_max)
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
