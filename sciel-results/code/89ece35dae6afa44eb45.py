import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # 1. Sieve prime factors up to 20000
    MAX_SUM = 20000
    prime_factors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        if not prime_factors[i]:
            for j in range(i, MAX_SUM + 1, i):
                prime_factors[j].append(i)
                
    # 2. Generate 2,3,5-smooth numbers <= 10000
    smooths = set()
    def gen_smooth(a, b, c):
        val = (2 ** a) * (3 ** b) * (5 ** c)
        if val > 10000:
            return
        smooths.add(val)
        if a < 14: gen_smooth(a+1, b, c)
        if b < 9: gen_smooth(a, b+1, c)
        if c < 6: gen_smooth(a, b, c+1)
    gen_smooth(0, 0, 0)
    pool = list(smooths)
    pool.sort()
    
    # 3. Precompute prime factor sets for all pairs in pool
    # Use a dictionary or 2D list
    # Since pool size is ~185, index mapping is easy
    n_pool = len(pool)
    pair_primes = [[None for _ in range(n_pool)] for _ in range(n_pool)]
    for i in range(n_pool):
        for j in range(i, n_pool):
            s = pool[i] + pool[j]
            # prime_factors[s] is a list, convert to frozenset for faster union
            pair_primes[i][j] = frozenset(prime_factors[s])
            pair_primes[j][i] = pair_primes[i][j]
            
    # 4. Local Search
    def calc_score(idx_set):
        # idx_set is a set of indices into pool
        # Union of pair_primes[i][j] for all i<j in idx_set
        all_primes = set()
        idx_list = list(idx_set)
        for k in range(len(idx_list)):
            i = idx_list[k]
            for m in range(k+1, len(idx_list)):
                j = idx_list[m]
                all_primes.update(pair_primes[i][j])
        return len(all_primes)
        
    best_score = float('inf')
    best_idx_set = None
    
    # Initial random sets
    for restart in range(50):
        if time.time() > deadline:
            break
        current_idx_set = set(random.sample(range(n_pool), 12))
        current_score = calc_score(current_idx_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try best improvement
            best_new_score = current_score
            best_new_set = None
            
            # Try swapping each element
            for i_idx in list(current_idx_set):
                # Try replacing with any other element not in set
                for j_idx in range(n_pool):
                    if j_idx in current_idx_set:
                        continue
                    new_set = set(current_idx_set)
                    new_set.remove(i_idx)
                    new_set.add(j_idx)
                    
                    new_score = calc_score(new_set)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_set = new_set
                        improved = True
                        # Don't break, keep looking for even better? 
                        # Or break to accept this improvement?
                        # Best-improvement usually finds better local minima.
                        # But it's slower.
                        # Given small size, let's keep looking for the best swap in this iteration.
            
            if best_new_set is not None:
                current_idx_set = best_new_set
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_idx_set = set(current_idx_set)
            else:
                improved = False
                
    if best_idx_set is None:
        # Fallback
        return list(range(1, 13))
        
    return [pool[i] for i in best_idx_set]

# EVOLVE-BLOCK-END
