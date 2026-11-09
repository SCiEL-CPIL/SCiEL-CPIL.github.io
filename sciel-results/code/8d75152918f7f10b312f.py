import time
import math
from itertools import combinations

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _prime_factors(s)
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr)
    
    # Try various initial constructions
    candidates = []
    
    # Powers of 2
    candidates.append([2**i for i in range(12)])
    
    # Small multiples of 30
    candidates.append([30*i for i in range(1, 13)])
    
    # Residue class mod small numbers
    for mod in range(2, 30):
        for res in range(mod):
            nums = [x for x in range(1, 10001) if x % mod == res]
            if len(nums) >= 12:
                candidates.append(nums[:12])
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(cand) < 12:
            continue
        cnt = _count_distinct_primes(cand)
        if cnt < best_count:
            best_count = cnt
            best_arr = list(cand)
    
    # Local search: try swapping elements
    def try_improve(arr, count):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            arr = list(arr)
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    # Try swapping arr[i] with a different value
                    # Try replacing arr[i] with arr[i] ± 1, ±2, etc.
                    for delta in [-3, -2, -1, 1, 2, 3, -10, -5, 5, 10, -20, -10, 10, 20, -50, -25, 25, 50]:
                        new_val = arr[i] + delta
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in arr and new_val != arr[i]:
                            continue
                        new_arr = list(arr)
                        new_arr[i] = new_val
                        cnt = _count_distinct_primes(new_arr)
                        if cnt < count:
                            count = cnt
                            arr = new_arr
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return arr, count
    
    best_arr, best_count = try_improve(best_arr, best_count)
    
    return best_arr
