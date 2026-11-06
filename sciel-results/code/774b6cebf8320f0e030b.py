import time
import random
import math

def _factorize(n):
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _prime_factor_union(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            primes |= _factorize(ai + arr[j])
    return primes

def construct():
    start = time.time()
    deadline = start + 75
    n = 12
    max_val = 10000
    random.seed(42)
    
    best = None
    best_cnt = float('inf')
    
    def try_set(s):
        nonlocal best, best_cnt
        if len(set(s)) != n or any(x < 1 or x > max_val for x in s):
            return False
        pf = _prime_factor_union(s)
        cnt = len(pf)
        if cnt < best_cnt:
            best_cnt = cnt
            best = list(s)
            return cnt <= 2
        return False
    
    # Structured candidates
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60]:
        s = [base * i for i in range(1, n + 1)]
        if all(x <= max_val for x in s) and try_set(s):
            return best
    
    for s in [list(range(2, 25, 2)), list(range(1, 25, 2)), list(range(3, 26, 3))]:
        if try_set(s):
            return best
    
    if best is None:
        best = list(range(1, n + 1))
        best_cnt = len(_prime_factor_union(best))
    
    current = list(best)
    temp = 100.0
    cool = 0.98
    
    while time.time() < deadline:
        arr = list(current)
        r = random.random()
        
        if r < 0.6:
            idx = random.randint(0, n - 1)
            old = arr[idx]
            arr[idx] = random.randint(1, max_val)
            if arr[idx] == old or arr.count(arr[idx]) > 1:
                continue
        elif r < 0.85:
            i, j = random.sample(range(n), 2)
            ni, nj = random.randint(1, max_val), random.randint(1, max_val)
            if ni in arr or nj in arr or ni == nj:
                continue
            arr[i], arr[j] = ni, nj
        else:
            arr = [random.randint(1, max_val) for _ in range(n)]
            if len(set(arr)) != n:
                continue
        
        pf = _prime_factor_union(arr)
        cnt = len(pf)
        
        if cnt < best_cnt:
            best_cnt = cnt
            best = list(arr)
            if cnt <= 2:
                break
            current = arr
            temp = 100.0
        else:
            delta = cnt - best_cnt
            if temp > 0.1 and random.random() < math.exp(-delta / temp):
                current = arr
        
        temp *= cool
    
    return best
