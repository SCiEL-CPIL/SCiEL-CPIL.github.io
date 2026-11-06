import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Pre-compute primes up to 10^6 for fast division of numbers up to 10^12
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
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
            result -= result // temp
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
        return k, seen

    best_k = 1
    best_s = 1
    
    max_n = 10**12

    # Strategy 1: Check small starts
    for s in range(1, 1000):
        if time.time() - start_time > DEADLINE:
            break
        k, seen = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            # Local extension
            while best_s > 1 and (time.time() - start_time < DEADLINE - 10):
                pv_left = phi(best_s - 1)
                if pv_left in seen:
                    break
                seen.add(pv_left)
                best_s -= 1
                best_k += 1
            while best_s + best_k - 1 < max_n and (time.time() - start_time < DEADLINE - 10):
                pv_right = phi(best_s + best_k)
                if pv_right in seen:
                    break
                seen.add(pv_right)
                best_k += 1

    # Strategy 2: Random search
    random.seed(42)
    for _ in range(50000):
        if time.time() - start_time > DEADLINE - 5:
            break
        s = random.randint(1, max_n - 200)
        k, seen = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            # Local extension
            while best_s > 1 and (time.time() - start_time < DEADLINE - 10):
                pv_left = phi(best_s - 1)
                if pv_left in seen:
                    break
                seen.add(pv_left)
                best_s -= 1
                best_k += 1
            while best_s + best_k - 1 < max_n and (time.time() - start_time < DEADLINE - 10):
                pv_right = phi(best_s + best_k)
                if pv_right in seen:
                    break
                seen.add(pv_right)
                best_k += 1

    return {'start': best_s, 'length': best_k}
