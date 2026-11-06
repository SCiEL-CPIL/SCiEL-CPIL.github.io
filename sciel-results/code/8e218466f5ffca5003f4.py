import math
import time

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Bases for n < 2^64
    # {2, 3, 5, 7, 11, 13, 17} is not enough for 2^64
    # {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} is safe for < 2^64?
    # Actually, for n < 3,825,123,056,546,413,051, the first 12 primes are sufficient.
    # Our n is < 10^18, which is < 3.8 * 10^18. So this is safe.
    for a in small_primes:
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
    if n % 5 == 0:
        return 5
    
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            # Restart with different c
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                # Try a different starting x
                x = 3
                y = 3
                c = 1
    return d

def factorize(n):
    """Returns a list of prime factors of n."""
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    
    d = pollard_rho(n)
    return factorize(d) + factorize(n // d)

def is_p_smooth(n, p, small_primes_list):
    """
    Checks if n is p-smooth (all prime factors <= p).
    Uses trial division for small p, Pollard Rho for large p.
    """
    if n == 1:
        return True
    
    # Quick check: if n is prime, it's p-smooth iff n <= p
    if is_prime(n):
        return n <= p
    
    # If p is small, use trial division
    # Threshold: if p < 10^5, trial division is faster
    if p < 100000:
        temp = n
        for q in small_primes_list:
            if q > p:
                break
            if q * q > temp:
                # If temp > 1, it's prime. Check if temp <= p
                if temp > 1:
                    return temp <= p
                else:
                    return True
            while temp % q == 0:
                temp //= q
        # If we exit loop, temp might be > 1
        if temp > 1:
            # temp is prime (since we checked up to sqrt(temp) or p)
            # Actually, if we broke because q > p, then temp has factor > p?
            # No, we removed all factors <= p.
            # So if temp > 1, it has a prime factor > p.
            return False
        return True
    else:
        # Use Pollard Rho
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 10^6
    limit = 10**6
    is_prime_sieve = [True] * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_sieve[j] = False
    
    primes = [i for i, prime in enumerate(is_prime_sieve) if prime]
    
    # We will iterate through primes
    # To be efficient, we can start from small primes
    # But we should also consider that larger p might be faster to reject?
    # No, larger p means larger n, slower factorization.
    # So small p are faster to process.
    
    # We'll iterate p in increasing order.
    # If time runs out, we return best.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Skip if p is too large for our time budget?
        # No, just let the loop break.
        
        # Check sequence
        k = 0
        # p^2 + 1
        # Optimization: if p^2+1 is prime, k=0
        n1 = p * p + 1
        if is_prime(n1):
            continue
            
        # Check smoothness for i=1 to 200
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p * p + i
            
            if is_p_smooth(n_val, p, primes):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 10^6
    limit = 10**6
    is_prime_sieve = [True] * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_sieve[j] = False
    
    primes = [i for i, prime in enumerate(is_prime_sieve) if prime]
    
    # We will iterate through primes
    # To be efficient, we can start from small primes
    # But we should also consider that larger p might be faster to reject?
    # No, larger p means larger n, slower factorization.
    # So small p are faster to process.
    
    # We'll iterate p in increasing order.
    # If time runs out, we return best.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check sequence
        k = 0
        # p^2 + 1
        # Optimization: if p^2+1 is prime, k=0
        n1 = p * p + 1
        if is_prime(n1):
            continue
            
        # Check smoothness for i=1 to 200
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p * p + i
            
            if is_p_smooth(n_val, p, primes):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
