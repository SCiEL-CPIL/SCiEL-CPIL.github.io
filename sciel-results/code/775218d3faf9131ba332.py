import time
import random
import math
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _prime_factors_of_sums(vals):
    """Return the union of prime factors of all pairwise sums of distinct elements."""
    n = len(vals)
    union = set()
    for i in range(n):
        for j in range(i + 1, n):
            s = vals[i] + vals[j]
            union |= _factorize(s)
    return union

def _score(vals):
    return len(_prime_factors_of_sums(vals))

def _initial_pool(max_val=10000, size=300):
    """Build a candidate pool biased toward small numbers and multiples of small primes."""
    pool = set()
    # Small numbers
    for x in range(1, 200):
        pool.add(x)
    # Multiples of small numbers up to max_val
    for base in range(2, 50):
        for x in range(base, max_val + 1, base):
            pool.add(x)
            if len(pool) >= size:
                break
        if len(pool) >= size:
            break
    # Fill with random
    while len(pool) < size:
        pool.add(random.randint(1, max_val))
    return list(pool)

def _local_search(vals, pool, deadline, max_iter=20000):
    """Swap one element with one from pool to reduce distinct prime count."""
    best = list(vals)
    best_score = _score(best)
    n = len(best)
    # Precompute pairwise sums and their prime factors for the current set
    def compute_pair_factors(arr):
        pf = {}
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                pf[(i, j)] = _factorize(s)
        union = set()
        for f in pf.values():
            union |= f
        return pf, union

    pf, union = compute_pair_factors(best)
    start = time.time()
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = random.randrange(n)
        new_val = random.choice(pool)
        if new_val in best:
            continue
        # Compute new union after swapping best[i] with new_val
        new_union = set()
        for (a, b), f in pf.items():
            if a == i or b == i:
                other = b if a == i else a
                s = new_val + best[other]
                new_union |= _factorize(s)
            else:
                new_union |= f
        if len(new_union) < best_score:
            best[i] = new_val
            best_score = len(new_union)
            pf, union = compute_pair_factors(best)
    return best, best_score

def _simulated_annealing(vals, pool, deadline, max_iter=50000):
    """SA with swap moves, allowing occasional uphill moves."""
    best = list(vals)
    best_score = _score(best)
    cur = list(best)
    cur_score = best_score
    n = len(cur)
    T = 2.0
    cooling = 0.99995
    start = time.time()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        i = random.randrange(n)
        new_val = random.choice(pool)
        if new_val in cur:
            continue
        # Compute new score
        new_cur = cur[:]
        new_cur[i] = new_val
        new_score = _score(new_cur)
        delta = new_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur = new_cur
            cur_score = new_score
            if cur_score < best_score:
                best = cur[:]
                best_score = cur_score
        T *= cooling
    return best, best_score

def construct():
    deadline = time.time() + 75  # 75s safety margin for 90s limit
    n = 12
    max_val = 10000

    # Build candidate pool
    pool = _initial_pool(max_val, size=400)
    
    best = list(range(1, 13))
    best_score = _score(best)
    
    # Phase 1: Multiple restarts with local search
    restarts = 30
    for r in range(restarts):
        if time.time() > deadline:
            break
        # Random initial set from pool
        cur = random.sample(pool, n)
        cur_score = _score(cur)
        if cur_score < best_score:
            best = cur[:]
            best_score = cur_score
        
        # Local search
        cur2, cur2_score = _local_search(cur, pool, deadline, max_iter=5000)
        if cur2_score < best_score:
            best = cur2
            best_score = cur2_score
    
    # Phase 2: SA on the best found so far
    if time.time() < deadline:
        best2, best2_score = _simulated_annealing(best, pool, deadline, max_iter=30000)
        if best2_score < best_score:
            best = best2
            best_score = best2_score
    
    # Phase 3: More local search on the best
    if time.time() < deadline:
        best3, best3_score = _local_search(best, pool, deadline, max_iter=10000)
        if best3_score < best_score:
            best = best3
            best_score = best3_score
    
    # Ensure all values are distinct and within bounds
    assert len(set(best)) == n
    assert all(1 <= x <= max_val for x in best)
    
    return best
