import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin

    # Precompute small primes up to 20000 (more coverage reduces Pollard-Rho calls)
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    small_primes = get_primes(20000)
    small_primes_set = set(small_primes)

    def is_prime(n):
        if n < 2:
            return False
        if n in small_primes_set:
            return True
        for p in small_primes:
            if p * p > n:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin for larger n
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        attempts = 0
        while True:
            attempts += 1
            if attempts > 20:
                # Try different polynomial
                c = random.randint(1, n - 1)
                x = random.randint(2, n - 1)
                y = x
            else:
                c = random.randint(1, n - 1)
                x = random.randint(2, n - 1)
                y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    phi_cache = {}

    def factor_large(n):
        # Returns list of prime factors of n (n > 1, not divisible by small primes)
        if n in phi_cache:
            return phi_cache[n]
        if is_prime(n):
            phi_cache[n] = [n]
            return [n]
        d = pollard_rho(n)
        result = factor_large(d) + factor_large(n // d)
        phi_cache[n] = result
        return result

    def phi(n):
        if n == 1:
            return 1
        if n in phi_cache:
            return phi_cache[n]
        result = n
        temp = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is either prime or product of two large primes
            if is_prime(temp):
                result -= result // temp
            else:
                factors = factor_large(temp)
                for p in set(factors):
                    result -= result // p
        phi_cache[n] = result
        return result

    def check_window(s):
        seen = set()
        length = 0
        for i in range(200):
            n = s + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            length += 1
        return length

    best_k = 1
    best_s = 1
    best_windows = []  # track top windows for local search

    # Strategy: random search + local refinement
    # Numbers with many distinct prime factors tend to have "unusual" totients
    # that are less likely to collide with neighbors

    def count_distinct_prime_factors(n, limit_primes=50):
        """Quick count of distinct prime factors using small primes only."""
        count = 0
        temp = n
        for p in small_primes[:limit_primes]:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count

    def local_search(s, max_steps=50):
        """Try to extend a window by shifting slightly."""
        best_local_k = 0
        best_local_s = s
        for offset in range(-max_steps, max_steps + 1):
            ns = s + offset
            if ns < 1 or ns + 200 > 10**12:
                continue
            k = check_window(ns)
            if k > best_local_k:
                best_local_k = k
                best_local_s = ns
        return best_local_k, best_local_s

    search_count = 0
    while time.time() < deadline:
        search_count += 1
        
        # Mix of strategies:
        # 1. Pure random (60%)
        # 2. Random + local search (30%)  
        # 3. Targeted: numbers with high prime factor diversity (10%)
        r = random.random()
        
        if r < 0.6:
            s = random.randint(10**10, 10**12 - 200)
            k = check_window(s)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 10:
                    best_windows.append((best_k, best_s))
                if best_k == 200:
                    break
        elif r < 0.9:
            # Random start, then local search
            s = random.randint(10**10, 10**12 - 200)
            k = check_window(s)
            if k >= 5:  # Only refine promising windows
                # Local search around this point
                for offset in range(-20, 21):
                    ns = s + offset
                    if ns < 1 or ns + 200 > 10**12:
                        continue
                    nk = check_window(ns)
                    if nk > k:
                        k = nk
                        s = ns
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 10:
                    best_windows.append((best_k, best_s))
                if best_k == 200:
                    break
        else:
            # Targeted search: pick numbers with many distinct prime factors
            # These tend to have "unique" totients
            for attempt in range(5):
                s = random.randint(10**10, 10**12 - 200)
                # Quick check: count distinct small prime factors of s
                df = count_distinct_prime_factors(s)
                if df >= 4:  # At least 4 distinct small prime factors
                    k = check_window(s)
                    if k > best_k:
                        best_k = k
                        best_s = s
                        if best_k >= 10:
                            best_windows.append((best_k, best_s))
                        if best_k == 200:
                            break

    # Final: if we found good windows, do one more pass of local search on best
    if best_k >= 8 and time.time() < deadline - 5:
        # Local search around the best window found
        for offset in range(-30, 31):
            ns = best_s + offset
            if ns < 1 or ns + 200 > 10**12:
                continue
            k = check_window(ns)
            if k > best_k:
                best_k = k
                best_s = ns

    return {'start': best_s, 'length': best_k}
