import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # Precompute small primes up to 10000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    small_primes = get_primes(10000)

    def is_prime(n):
        if n < 2:
            return False
        for p in small_primes:
            if n % p == 0:
                return n == p
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

    def factor_large(n):
        # Returns list of prime factors of n (n > 1, not divisible by small primes)
        if is_prime(n):
            return [n]
        d = pollard_rho(n)
        return factor_large(d) + factor_large(n // d)

    def phi(n):
        if n == 1:
            return 1
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
            factors = factor_large(temp)
            for p in set(factors):
                result -= result // p
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

    # Search
    while time.time() < deadline:
        # Random start in [10^10, 10^12 - 200]
        s = random.randint(10**10, 10**12 - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
