import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    # Pre-compute primes up to 10^6 for fast trial division
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    primes_set = set(primes)

    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
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

    def extend_left(s, current_len, max_n=10**12):
        """Try to extend the run to the left."""
        seen = set()
        # First collect all phi values in current window
        for i in range(current_len):
            seen.add(phi(s + i))
        # Try extending left
        new_s = s
        for back in range(1, 50):
            v = s - back
            if v < 1:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            new_s = v
        return new_s, new_s and (s - new_s) + current_len or current_len

    best_k = 1
    best_s = 1
    max_n = 10**12

    # Strategy 1: Check small starts
    for s in range(1, 5000):
        if time.time() - start_time > DEADLINE:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Search near primorials and smooth numbers
    primorial_values = [1]
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if primorial_values[-1] * p > max_n:
            break
        primorial_values.append(primorial_values[-1] * p)

    for pm in primorial_values:
        if time.time() - start_time > DEADLINE - 30:
            break
        for mult in range(1, 20):
            s = pm * mult
            if s > max_n - 200 or s < 1:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            # Try nearby offsets
            for delta in range(-50, 51, 5):
                if time.time() - start_time > DEADLINE - 30:
                    break
                ns = s + delta
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

    # Strategy 3: Structured search in mid-range
    for base in range(10**4, 10**8, 10**4):
        if time.time() - start_time > DEADLINE - 20:
            break
        for offset in range(-100, 101, 10):
            s = base + offset
            if s < 1 or s > max_n - 200:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s

    # Strategy 4: Random search with bias
    random.seed(42)
    for _ in range(3000):
        if time.time() - start_time > DEADLINE - 10:
            break
        r = random.random()
        if r < 0.2:
            s = random.randint(1, 10**6)
        elif r < 0.5:
            s = random.randint(10**6, 10**9)
        else:
            s = random.randint(10**9, max_n - 200)
        s = max(1, min(s, max_n - 200))
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 5: Polish - extend best found run
    if time.time() - start_time < DEADLINE - 5:
        # Try extending left and right from best_s
        new_s, new_k = extend_left(best_s, best_k)
        if new_k > best_k:
            best_k = new_k
            best_s = new_s

        # Try extending right
        end = best_s + best_k - 1
        seen = set()
        for i in range(best_k):
            seen.add(phi(best_s + i))
        for fwd in range(1, 50):
            v = end + fwd
            if v > max_n:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            best_k += 1

    return {'start': best_s, 'length': best_k}
