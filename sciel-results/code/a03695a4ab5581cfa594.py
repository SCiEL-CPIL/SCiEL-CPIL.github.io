import random
from itertools import combinations

def construct():
    # Design from scratch: use a greedy/local-search approach over a rich candidate pool.
    # The key insight: we want a set A of 20 distinct positive integers <= 10000 that minimizes
    # max(|A+A|, |A*A|). 
    # 
    # Strategy:
    # 1. Build a candidate pool of "structured" numbers that tend to have many additive/multiplicative collisions.
    # 2. Use a greedy construction with local search (swap-based) to minimize the objective.
    #
    # Candidate pool: include small integers, powers, and numbers with many small prime factors (highly composite-like).
    # Numbers with many divisors tend to produce more product collisions. Small numbers produce more sum collisions.
    
    # Build candidate pool
    pool = set()
    
    # Small integers 1..60
    for i in range(1, 61):
        pool.add(i)
    
    # Powers of 2, 3, 5, 6, 7, 8, 9, 10 up to 10000
    for base in [2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 48, 54, 64, 72, 80, 96, 108, 128, 144, 160, 192, 216, 256, 288, 320, 384, 432, 512, 576, 640, 768, 864, 1024, 1152, 1280, 1536, 1728, 2048, 2304, 2560, 3072, 3456, 4096, 4608, 5120, 6144, 6912, 8192, 9216, 10240, 12288, 13824, 16384, 18432, 20480, 24576, 27648, 32768, 36864, 40960, 49152, 55296, 65536, 73728, 81920, 98304, 110592, 131072, 147456, 163840, 196608, 221184, 262144, 294912, 327680, 393216, 442368, 524288, 589824, 655360, 786432, 884736, 1048576]:
        p = base
        while p <= 10000:
            pool.add(p)
            p *= base
    
    # 2^i * 3^j (Hamming 2,3-smooth)
    for i in range(15):
        for j in range(10):
            v = (2 ** i) * (3 ** j)
            if v <= 10000:
                pool.add(v)
    
    # 2^i * 3^j * 5^k (5-smooth)
    for i in range(15):
        for j in range(10):
            for k in range(8):
                v = (2 ** i) * (3 ** j) * (5 ** k)
                if v <= 10000:
                    pool.add(v)
    
    # 2^i * 3^j * 5^k * 7^l (7-smooth) - limited
    for i in range(14):
        for j in range(9):
            for k in range(7):
                for l in range(5):
                    v = (2 ** i) * (3 ** j) * (5 ** k) * (7 ** l)
                    if v <= 10000:
                        pool.add(v)
    
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    
    # Objective function
    def score(A):
        A = list(A)
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    # Initial set: start with 20 smallest from pool
    best_set = pool[:20]
    best_score = score(best_set)
    
    # Local search: try swaps
    random.seed(42)
    improved = True
    max_iters = 5000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try random swaps
        for _ in range(200):
            idx_in = random.randint(0, len(best_set) - 1)
            # Pick a random candidate from pool not in best_set
            cand = random.choice(pool)
            if cand in best_set:
                continue
            
            new_set = list(best_set)
            new_set[idx_in] = cand
            new_set.sort()
            
            s = score(new_set)
            if s < best_score:
                best_score = s
                best_set = new_set
                improved = True
                break
    
    # Also try a more systematic local search: for each element, try replacing with nearby pool elements
    improved = True
    while improved:
        improved = False
        for idx in range(len(best_set)):
            old_val = best_set[idx]
            # Try replacing with elements near old_val in pool
            pool_idx = pool.index(old_val) if old_val in pool else 0
            for delta in range(-10, 11):
                new_idx = pool_idx + delta
                if 0 <= new_idx < len(pool):
                    cand = pool[new_idx]
                    if cand in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[idx] = cand
                    new_set.sort()
                    s = score(new_set)
                    if s < best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
    
    return best_set
