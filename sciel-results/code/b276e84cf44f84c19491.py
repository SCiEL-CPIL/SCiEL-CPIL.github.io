import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Precompute primes up to 10^6 for better coverage
    P_MAX = 1000000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    
    def is_p_smooth_fast(n, p, primes_list):
        if n <= 1:
            return True
        rem = n
        for q in primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
            if rem == 1:
                return True
        if rem == 1:
            return True
        if rem <= p:
            return True
        return False

    def find_max_k(p, max_k=200):
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth_fast(p2 + j, p, primes):
                break
            k += 1
        return k

    best_k = 1
    best_p = 2

    # Structural change: Two-phase search with quick rejection
    # Phase 1: Quick screen primes in [50000, 1000000] by checking only p^2+1
    # Phase 2: Full check for survivors
    # This avoids spending time on primes that fail immediately
    
    # Phase 1: Quick screen
    candidates = []
    for p in reversed(primes):
        if p < 50000:
            break
        if time.time() - start_time > DEADLINE:
            break
        p2 = p * p
        if is_p_smooth_fast(p2 + 1, p, primes):
            candidates.append(p)
    
    # Phase 2: Full check on candidates
    for p in candidates:
        if time.time() - start_time > DEADLINE:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 3: If time remains, check smaller primes
    for p in reversed(primes):
        if p >= 50000:
            continue
        if time.time() - start_time > DEADLINE:
            break
        p2 = p * p
        if not is_p_smooth_fast(p2 + 1, p, primes):
            continue
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
