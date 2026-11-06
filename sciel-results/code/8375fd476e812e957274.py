import math
import time
import random

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Deterministic witnesses for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    
    x = random.randrange(2, n)
    y = x
    c = random.randrange(1, n)
    d = 1
    
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        
    if d != n:
        return d
    else:
        return pollard_rho(n)

def factorize(n):
    factors = []
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

def _gen_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i] and is_prime(i)]

# We need primes for trial division.
# For p up to 10^9, we can't generate primes up to p.
# But for p-smoothness check, we only need to check if n has any prime factor > p.
# If n < p^2, and n is not p-smooth, then n has a prime factor q > p.
# Since n < p^2, q < n < p^2. Also n/q < p.
# So n = q * m where q > p and m < p.
# If we trial divide by primes up to p, we would find q if q <= p, but q > p.
# So we must find q.
# Pollard Rho is good for finding factors.
# But we need to check if ALL factors are <= p.
# So we factorize n, and check max(factors) <= p.
# This is expensive if we factorize every number.

# Strategy:
# 1. Precompute primes up to 10^6 for trial division.
# 2. For a candidate p, check p^2 + 1, ..., p^2 + 200.
# 3. For each n, first trial divide by primes <= 10^6.
# 4. If remaining factor > 1, check if it's prime.
#    If prime, check if <= p.
#    If composite, use Pollard Rho to find a factor.
#    Recursively check factors.
#    Actually, just factorize completely and check max factor.

# Optimization:
# If p is large, most small numbers n = p^2 + i will have small factors.
# We can quickly rule out many candidates.

MAX_TRIAL_PRIME = 100_000
PRIMES = _gen_primes(MAX_TRIAL_PRIME)

def is_p_smooth(n, p):
    if n <= p:
        return True
    
    # Trial division by small primes
    m = n
    for q in PRIMES:
        if q > p:
            break
        if q * q > m:
            break
        while m % q == 0:
            m //= q
        if m == 1:
            return True
        if m <= p:
            # If m <= p, it's p-smooth (since we've divided out all small factors, m is either 1 or prime > max_trial_prime or composite with large factors... wait)
            # If m <= p, and we've divided by all primes q <= min(p, sqrt(m)), then m is prime.
            # If m is prime and m <= p, it's smooth.
            # If m is composite, it has a factor <= sqrt(m) <= sqrt(p).
            # We divided by all primes <= sqrt(m) (since q*q > m check).
            # So m must be 1 or prime.
            # So if m <= p, it is p-smooth.
            return True
            
    # If we get here, m > 1 and m has no prime factors <= min(p, sqrt(original m roughly))
    # Actually, we broke loop because q > p or q*q > m.
    # If q > p: we checked all primes <= p.
    #   If m > 1, m has no prime factors <= p.
    #   So m is not p-smooth (unless m=1).
    #   Return False.
    # If q*q > m: we checked all primes <= sqrt(m).
    #   So m is prime.
    #   Return m <= p.
    
    # How to distinguish?
    # The loop breaks if q > p OR q*q > m.
    # If q > p, we stop.
    # If q*q > m, we stop.
    
    # Let's restructure.
    
    # Actually, the above loop logic is slightly flawed for the "q > p" case if m is composite with factors > p.
    # But if m has a factor > p, and m < p^2 (which is true for n = p^2 + i, i <= 200, for p > 200),
    # then m = a * b where a, b > p implies m > p^2. Contradiction.
    # So m can have at most one prime factor > p.
    # If m has a prime factor q > p, then m = q * r where r < p.
    # We would have divided out r if r is composed of primes <= MAX_TRIAL_PRIME.
    # If r has a prime factor > MAX_TRIAL_PRIME, we might not have divided it out.
    
    # Let's just factorize m completely if trial division doesn't finish it.
    
    if m == 1:
        return True
    
    # Check if m is prime
    if is_prime(m):
        return m <= p
    
    # m is composite. Factorize it.
    factors = factorize(m)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Search primes p.
    # We can't iterate all primes up to 10^9.
    # Let's search in ranges.
    # Small p: p^2 is small, numbers are small.
    # Large p: p^2 is large.
    
    # Let's try a few specific ranges.
    # 1. Primes up to 10^6 (we have them)
    # 2. Random primes in [10^6, 10^9]
    
    # First, check primes we have
    for p in PRIMES:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            if is_p_smooth(n_val, p):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k == 200:
            break

    # If we haven't found k=200, try larger primes.
    # Generate random primes in [10^6, 10^9]
    # This is a heuristic search.
    
    # To find random primes, we can pick random odd numbers and test primality.
    # But we need to be efficient.
    
    # Let's try a few hundred random primes.
    # Actually, the probability of a long run might be low.
    # But the task says "Long runs for p <= 10^9 are finite evidence...".
    
    # Let's stick to the precomputed primes for now, as they are the most reliable.
    # If time remains, we can try larger ones.
    
    # Wait, the current program only checks primes up to 10^7 (in the seed, it was 10^7, here I used 10^5 for trial division primes, but PRIMES list is only up to 10^5).
    # I should generate more primes for the search space, not just for trial division.
    
    # Let's generate primes up to 10^7 for the search space.
    # And use primes up to 10^5 for trial division.
    
    # Re-initialize
    SEARCH_PRIMES = _gen_primes(10_000_000)
    
    best_k = 0
    best_p = 2
    
    for p in SEARCH_PRIMES:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            if is_p_smooth(n_val, p):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
