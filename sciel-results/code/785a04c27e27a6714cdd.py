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

    # --- Phase 1: NumPy Sieve for Small/Medium Range ---
    # Use a larger range to find longer runs of distinct totients
    
    # First, determine how much time we have for the sieve
    t0 = now()
    
    # Try progressively larger sieves until we run out of time
    # Start with 10M, can go up to 50M if fast enough
    N = 10 * 10**6  # 10 million
    
    try:
        # Sieve of Eratosthenes to get primes, then compute phi via sieve method
        phi_arr = np.arange(N + 1, dtype=np.int64)
        
        # Standard totient sieve: for each prime p, update multiples
        # Only iterate i where phi[i] == i (i is prime)
        # Optimize by only checking up to sqrt(N) for primes
        sqrt_N = int(math.isqrt(N))
        for i in range(2, N + 1):
            if phi_arr[i] == i:  # i is prime
                # Update all multiples of i
                phi_arr[i::i] -= phi_arr[i::i] // i
            
            if i > sqrt_N and (i % 1000000 == 0):
                if now() > DEADLINE - 40.0:
                    # If taking too long, break and use what we have
                    # But phi would be incomplete, so we need to handle this
                    # For now, just let it finish if close to deadline
                    break

        # Find max distinct run
        vals = phi_arr[1:]
        n_vals = len(vals)
        
        k, idx = max_distinct_run(vals, 200)
        if k > best_k:
            best_k = k
            best_s = idx + 1
            
    except Exception as e:
        # Fallback to smaller N
        N = 2 * 10**6
        try:
            phi_arr = np.arange(N + 1, dtype=np.int64)
            for i in range(2, N + 1):
                if phi_arr[i] == i:
                    phi_arr[i::i] -= phi_arr[i::i] // i
            vals = phi_arr[1:]
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = idx + 1
        except:
            pass

    # --- Phase 2: Targeted Search in Larger Ranges ---
    # Strategy: Check specific ranges where long runs might occur
    # Based on number theory, longer runs of distinct totients are more likely
    # when numbers are large and have varied factorizations
    
    if now() < DEADLINE - 10.0:
        random.seed(42)
        
        # Define search regions: mix of structured and random
        # 1. Powers of 10 neighborhoods
        # 2. Random large ranges
        # 3. Ranges with many primes nearby (high chance of distinct phis)
        
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
        
        # Sort by descending priority (larger numbers first)
        search_regions.sort(reverse=True)
        
        # Process each region
        for s in search_regions:
            if now() > DEADLINE - 5.0:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            
            # Compute phi for 200 consecutive numbers
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
        # Search in a window around the best start found
        window = 5000
        s_min = max(1, best_s - window)
        s_max = min(10**12 - 200, best_s + window)
        
        # Sample some points in this window - use systematic sampling
        n_samples = 50
        for i in range(n_samples):
            if now() > DEADLINE - 2.0:
                break
            # Systematic sampling across the window
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
