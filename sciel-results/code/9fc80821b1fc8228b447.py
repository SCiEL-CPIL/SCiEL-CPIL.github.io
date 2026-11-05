import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    # Precompute primes
    SIEVE_LIMIT = 10_000_000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    primes_list = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    # Miller-Rabin for 64-bit
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in bases:
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

    def is_p_smooth(n, p, primes_list):
        if n <= p:
            return True
        if is_prime_mr(n):
            return False
        
        temp = n
        for q in primes_list:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        if temp == 1:
            return True
        if temp <= p:
            return True
        return False

    best_k = 1
    best_p = 2

    for p in primes_list:
        if time.time() - start_time > DEADLINE:
            break
        
        k = 0
        p2 = p * p
        for j in range(1, 201):
            n = p2 + j
            if is_p_smooth(n, p, primes_list):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
