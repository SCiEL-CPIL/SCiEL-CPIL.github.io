import random
import time
import math

def construct():
            # Sieve of Eratosthenes up to 10^6
            limit = 1000000
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            primes = [i for i, val in enumerate(is_prime) if val]
            
            def phi(n):
                if n == 1:
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
                return result
            
            best_k = 1
            best_s = 1
            
            # Phase 1: Small exhaustive search
            # Up to 100,000
            for s in range(1, 100000):
                seen = set()
                k = 0
                while k < 200:
                    v = phi(s + k)
                    if v in seen:
                        break
                    seen.add(v)
                    k += 1
                if k > best_k:
                    best_k = k
                    best_s = s
            
            # Phase 2: Random search
            start_time = time.time()
            time_limit = 115
            
            # We'll try to do as many trials as possible
            # Use random log-uniform distribution
            while time.time() - start_time < time_limit:
                # Pick random start between 10^6 and 10^12
                # log10(s) in [6, 12]
                log_s = random.uniform(6, 12)
                s = int(10 ** log_s)
                
                # Ensure s is within bounds
                if s < 1 or s + 200 > 10**12:
                    continue
                
                seen = set()
                k = 0
                while k < 200:
                    v = phi(s + k)
                    if v in seen:
                        break
                    seen.add(v)
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_s = s
                    if best_k == 200:
                        break
            
            return {'start': best_s, 'length': best_k}
