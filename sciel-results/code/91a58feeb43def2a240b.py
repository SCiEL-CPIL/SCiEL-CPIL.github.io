"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    # sqrt(10^12) = 10^6, so primes up to 10^6 are sufficient for factorization
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)
    prime_set = set(primes)

    def is_prime_large(n):
        """Check if n is prime using precomputed primes up to 10^6."""
        if n < 2:
            return False
        if n < 10**6 + 1:
            return n in prime_set
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                return False
        return True

    def find_prime_near(n, direction=-1, max_search=10000):
        """Find a prime near n in the given direction (-1 for down, 1 for up)."""
        if direction < 0:
            for i in range(max_search):
                candidate = n - i
                if candidate < 2:
                    return None
                if is_prime_large(candidate):
                    return candidate
        else:
            for i in range(max_search):
                candidate = n + i
                if is_prime_large(candidate):
                    return candidate
        return None

    phi_cache = {}

    def phi_fast(n):
        if n in phi_cache:
            return phi_cache[n]
        if n == 1:
            phi_cache[n] = 1
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        phi_cache[n] = result
        return result

    def check_run(s, max_k=200):
        """Check if phi(s), ..., phi(s+max_k-1) are distinct. Returns length of distinct prefix."""
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Strategy: Start at a large prime to minimize collision probability.
    # phi(prime) = prime - 1, which is large and unique.
    # Target near 10^12 to maximize distinctness.
    
    # Find a prime near 10^12 - 200 (leaving room for 200 numbers)
    target = 10**12 - 200
    base_s = find_prime_near(target, direction=-1, max_search=5000)
    
    if base_s is None or base_s < 1:
        base_s = target

    best_k = 1
    best_s = base_s

    # First, check the prime-anchored run
    k = check_run(base_s, 200)
    if k > best_k:
        best_k = k
        best_s = base_s

    if best_k == 200:
        return {'start': best_s, 'length': best_k}

    # If not 200, try nearby primes and offsets
    # Search in a window around our best candidate
    search_window = 10000
    found = False
    
    for offset in range(1, search_window):
        if time_left() < 10:
            break
        
        # Try primes in the window
        for direction in [-1, 1]:
            s = base_s + direction * offset
            if s < 1 or s + 199 > 10**12:
                continue
            
            if is_prime_large(s):
                k = check_run(s, 200)
                if k > best_k:
                    best_k = k
                    best_s = s
                    if best_k == 200:
                        found = True
                        break
        
        if found:
            break

    if best_k == 200:
        return {'start': best_s, 'length': best_k}

    # Broader search if needed
    if time_left() > 20:
        # Try random offsets in a wider range
        import random
        random.seed(42)
        for _ in range(100):
            if time_left() < 10:
                break
            
            # Random start in [10^11, 10^12 - 200]
            s = random.randint(10**11, 10**12 - 200)
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k == 200:
                    break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
