import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Pre-compute primes up to 10^6 for fast division
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Pre-compute primorial-like numbers (products of first k primes)
    # These are highly composite and create regions where phi values vary widely
    primorials = [2]
    for p in primes:
        if primorials[-1] * p > 10**12:
            break
        primorials.append(primorials[-1] * p)
    # Also compute some scaled versions
    interesting_starts = set()
    for pm in primorials:
        for mult in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            s = pm * mult
            if 1 <= s <= 10**12 - 200:
                interesting_starts.add(s)
            # Also try s near pm
            for offset in range(-50, 51):
                s2 = pm + offset
                if 1 <= s2 <= 10**12 - 200:
                    interesting_starts.add(s2)
    # Add some structured candidates: LCM of 1..n for small n
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    lcms = [1]
    for i in range(2, 30):
        lcms.append(lcm(lcms[-1], i))
        if lcms[-1] > 10**12:
            lcms.pop()
            break
    for lc in lcms:
        for mult in [1, 2, 3, 5, 7]:
            s = lc * mult
            if 1 <= s <= 10**12 - 200:
                interesting_starts.add(s)
            for offset in range(-30, 31):
                s2 = lc + offset
                if 1 <= s2 <= 10**12 - 200:
                    interesting_starts.add(s2)
    interesting_starts = sorted(interesting_starts)
    
    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2: return False
        # Check small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p: return True
            if n % p == 0: return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's rho for factoring composites
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

    def factorize(n):
        """Full factorization using trial division + Pollard's rho"""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime_mr(n):
                factors.append(n)
            else:
                # Use Pollard's rho
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m == 1:
                        continue
                    if is_prime_mr(m):
                        factors.append(m)
                    else:
                        d = pollard_rho(m)
                        if d == m:
                            # Retry
                            stack.append(m)
                        else:
                            stack.append(d)
                            stack.append(m // d)
        return factors

    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        # Divide by small primes
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # Use Pollard's rho to factor the composite remainder
                facs = factorize(temp)
                for p in set(facs):
                    result -= result // p
        return result

    def check_window(s, max_len=200, max_n=10**12):
        seen = set()
        k = 0
        for i in range(max_len):
            v = s + i
            if v > max_n:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            k += 1
        return k

    best_k = 1
    best_s = 1
    
    max_n = 10**12

    # Strategy 1: Check small starts
    for s in range(1, 500):
        if time.time() - start_time > DEADLINE - 5:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Structured search around primorials and LCMs
    # These are regions where phi values tend to be large and varied
    for s in interesting_starts:
        if time.time() - start_time > DEADLINE - 5:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
        # Also check nearby starts
        for offset in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]:
            s2 = s + offset
            if 1 <= s2 <= max_n - 200 and time.time() - start_time < DEADLINE - 5:
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_s = s2

    # Strategy 3: Search in arithmetic progressions near highly composite numbers
    # Try windows starting at n where n is close to a multiple of a large primorial
    for base in primorials:
        if base > 10**10:
            break
        # Search in a window around multiples of base
        for mult in range(1, min(100, int(10**12 / base) + 1)):
            s0 = base * mult
            if s0 > max_n - 200:
                break
            for offset in range(0, 20, 2):
                s = s0 + offset
                if s > max_n - 200:
                    break
                if time.time() - start_time > DEADLINE - 5:
                    break
                k = check_window(s, 200)
                if k > best_k:
                    best_k = k
                    best_s = s

    # Strategy 4: Focused random search near interesting regions
    random.seed(42)
    if interesting_starts:
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 3:
                break
            # Pick a random interesting start and add a random offset
            base_s = random.choice(interesting_starts)
            offset = random.randint(-1000, 1000)
            s = base_s + offset
            if 1 <= s <= max_n - 200:
                k = check_window(s, 200)
                if k > best_k:
                    best_k = k
                    best_s = s

    return {'start': best_s, 'length': best_k}
