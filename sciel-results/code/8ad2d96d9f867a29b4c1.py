# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
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
                x = x * x % n
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
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = abs(x - y)
                if d == 0:
                    break
            if d != n:
                return d
    
    def factor(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = pollard_rho(n)
        return factor(d) + factor(n // d)
    
    def phi(x):
        if x == 1:
            return 1
        factors = {}
        for p in factor(x):
            factors[p] = factors.get(p, 0) + 1
        result = x
        for p in factors:
            result -= result // p
        return result
    
    def find_best_in_range(start, end):
        """Find best run in [start, end], return (length, start_pos)"""
        if start > end:
            return (0, start)
        n = end - start + 1
        phis = [0] * n
        for i in range(n):
            if time.time() - start_time > DEADLINE - 3:
                return (0, start)
            phis[i] = phi(start + i)
        
        best = (0, start)
        seen = set()
        k = 0
        for i in range(n):
            if phis[i] not in seen:
                seen.add(phis[i])
                k += 1
            else:
                if k > best[0]:
                    best = (k, start + i - k)
                seen.clear()
                seen.add(phis[i])
                k = 1
        if k > best[0]:
            best = (k, start + n - k)
        return best
    
    # Quick baseline search
    best = find_best_in_range(1, 10000)
    
    # Search in progressively larger ranges
    # Keep ranges small enough to be feasible
    ranges = [
        (10000, 50000),
        (50000, 150000),
        (150000, 400000),
        (400000, 1000000),
        (1000000, 3000000),
        (3000000, 10000000),
        (10000000, 30000000),
        (30000000, 100000000),
        (100000000, 300000000),
        (300000000, 1000000000),
        (1000000000, 3000000000),
        (3000000000, 10000000000),
    ]
    
    for (s, e) in ranges:
        if time.time() - start_time > DEADLINE - 5:
            break
        if e > 10**12:
            e = 10**12
        result = find_best_in_range(s, e)
        if result[0] > best[0]:
            best = result
    
    return {'start': best[1], 'length': best[0]}
# EVOLVE-BLOCK-END
