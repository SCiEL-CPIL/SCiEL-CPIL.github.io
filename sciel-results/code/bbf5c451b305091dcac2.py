import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
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

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_k = 0
    best_p = 3

    # We established that for p > 200, p^2+j is p-smooth iff it is composite.
    # So we just need to find a prime p such that p^2+1 ... p^2+k are all composite.
    
    # Search strategy:
    # 1. Iterate through primes p.
    # 2. For each p, check j=1..200 for compositeness.
    # 3. Keep the max k.
    
    # To speed up, we can skip p where p^2+1 is prime (prob ~1/ln(p^2)).
    # We'll generate primes up to some limit, then maybe random sample larger ones if needed.
    
    # Let's try primes up to 10^6 first, as they are easy to sieve.
    # If we find k=200, we can stop early or try to verify with larger p?
    # The task says "Maximize k". k is capped at 200. So if we find k=200, we are done (score 200).
    # If we don't find 200, we return the best found.
    
    # Sieve primes up to 10^6
    limit = 10**6
    primes = _sieve(limit)
    
    # We can also try some larger primes randomly if we have time.
    # But let's see how fast the sieve check is.
    # Check for each p:
    # for j in 1..200: if is_prime(p^2+j): break
    
    # Optimization: If p^2+1 is prime, k=0. We can skip the rest.
    
    for p in primes:
        if time.time() > deadline:
            break
        if p <= 200:
            continue # Must have p > 200 for the equivalence to hold strictly without edge cases, though technically small p works too but the logic "composite implies smooth" relies on j<p.
        
        p_sq = p * p
        k = 0
        for j in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + j
            # Check if val is prime. If prime, it's NOT p-smooth.
            # If composite, it IS p-smooth (given j < p).
            if _is_prime_mr(val):
                break
            else:
                k = j
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                # Found max possible k. Can return immediately?
                # The task says "Maximize k". 200 is the cap.
                # We should probably return this.
                return {'p': best_p, 'k': best_k}

    # If we didn't find 200 in small primes, try larger primes randomly.
    # The probability of a long run of composites is independent of p (roughly).
    # But larger p means larger numbers for Miller-Rabin, which is slightly slower.
    # However, we might have missed a good p in the gap between 10^6 and 10^9?
    # Actually, the distribution of "runs of composites" in p^2+j should be similar.
    # Let's try random sampling of primes in [10^6, 10^9].
    
    attempts = 0
    while time.time() < deadline and attempts < 10000:
        attempts += 1
        # Random number in [10^6, 10^9]
        p = random.randint(10**6, 10**9)
        if p % 2 == 0:
            p += 1
        # Find next prime
        while not _is_prime_mr(p):
            p += 2
            if p > 10**9:
                break
        if p > 10**9 or not _is_prime_mr(p):
            continue
        
        p_sq = p * p
        k = 0
        for j in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + j
            if _is_prime_mr(val):
                break
            else:
                k = j
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
