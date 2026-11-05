import time
import random
import math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 80.0  # safety margin below 90s kill

    N = 12
    MAXV = 10000

    # 1. Precompute prime factors for all numbers up to 2*MAXV
    limit = 2 * MAXV
    # spf: smallest prime factor
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Precompute factor sets for all possible sums (2 to 2*MAXV)
    # Using frozenset for hashing and fast union operations
    sum_factors = [frozenset() for _ in range(limit + 1)]
    for s in range(2, limit + 1):
        sum_factors[s] = get_prime_factors(s)

    # Cost function: number of distinct prime factors in the union of all pairwise sums
    # To speed up, we can maintain the union incrementally, but for N=12, full recalculation is 66 pairs.
    # 66 set unions of small frozensets is fast.
    def cost(arr):
        union = set()
        for i in range(N):
            ai = arr[i]
            for j in range(i + 1, N):
                s = ai + arr[j]
                union.update(sum_factors[s])
        return len(union)

    # Strategy:
    # The key insight from inspiration 1 (score 1.1250) is "prime-restriction".
    # If we choose a set of primes P, and want all sums to be P-smooth,
    # we can try to construct numbers such that their sums fall into the set of P-smooth numbers.
    
    # Let's try a constructive approach first, then refine with local search.
    
    # Hypothesis: If we pick numbers from a specific arithmetic progression or structure,
    # the sums might be constrained to a small set of prime factors.
    
    # A very strong baseline: all numbers are multiples of a highly composite number?
    # If all a_i are multiples of M, then a_i + a_j is a multiple of M.
    # So all prime factors of M are in the union.
    # Plus, the quotients (a_i+ a_j)/M must be smooth.
    
    # Let's try to find a set of 12 numbers such that all pairwise sums are 2^a * 3^b * 5^c ...
    # Actually, let's just run a more effective local search/SA.
    
    # Improved Local Search:
    # 1. Start with a good initial guess.
    # 2. Use a move that replaces one element with a "better" candidate.
    #    Instead of random, we can score candidates by how many "new" primes they introduce.
    
    def local_search(initial, max_iters=5000):
        cur = initial[:]
        cur_cost = cost(cur)
        best = cur[:]
        best_cost = cur_cost
        
        # Precompute for each position, the contribution of that element to the union?
        # That's complex. Let's stick to full cost evaluation but optimize the loop.
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Pick a random index to replace
            idx = random.randint(0, N - 1)
            old_val = cur[idx]
            
            # Generate a few candidate replacements
            # 1. Random
            # 2. Neighbors of old_val
            # 3. Values that make sums with other elements "smooth"
            
            candidates = set()
            candidates.add(random.randint(1, MAXV))
            candidates.add(max(1, old_val + random.randint(-10, 10)))
            candidates.add(max(1, old_val + random.randint(-100, 100)))
            
            # Heuristic: Try values that are multiples of small primes
            for m in [2, 3, 5, 6, 10, 12, 30, 60]:
                k = random.randint(1, MAXV // m)
                candidates.add(k * m)
                
            best_new_val = None
            best_new_cost = cur_cost
            
            for new_val in candidates:
                if new_val < 1 or new_val > MAXV:
                    continue
                # Check distinctness
                if new_val != old_val and new_val in cur:
                    continue
                    
                new_arr = cur[:]
                new_arr[idx] = new_val
                
                # Quick check: if new_val is the same as another, skip (already handled by set check above roughly)
                # But we need to be sure.
                if len(set(new_arr)) != N:
                    continue
                
                nc = cost(new_arr)
                if nc < best_new_cost:
                    best_new_cost = nc
                    best_new_val = new_val
            
            if best_new_val is not None:
                cur[idx] = best_new_val
                cur_cost = best_new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best = cur[:]
                    
            # If we stuck for a while, try a random jump?
            # For now, just keep trying.
            
        return best, best_cost

    # Initial solutions to try
    # 1. Multiples of 2
    # 2. Multiples of 3
    # 3. Multiples of 6
    # 4. Random
    # 5. "Smooth" numbers (powers of 2, 3, 5)
    
    inits = []
    
    # Multiples of 2
    inits.append(random.sample(range(2, MAXV + 1, 2), N))
    # Multiples of 3
    inits.append(random.sample(range(3, MAXV + 1, 3), N))
    # Multiples of 6
    inits.append(random.sample(range(6, MAXV + 1, 6), N))
    # Multiples of 10
    inits.append(random.sample(range(10, MAXV + 1, 10), N))
    # Multiples of 30
    inits.append(random.sample(range(30, MAXV + 1, 30), N))
    
    # Powers of 2, 3, 5 combinations
    # Generate a pool of 5-smooth numbers
    smooth_pool = []
    for a in range(15):
        for b in range(10):
            for c in range(7):
                v = (2**a) * (3**b) * (5**c)
                if v <= MAXV:
                    smooth_pool.append(v)
    if len(smooth_pool) >= N:
        inits.append(random.sample(smooth_pool, N))
    
    # Random
    inits.append(random.sample(range(1, MAXV + 1), N))
    
    best_global = None
    best_global_cost = float('inf')
    
    for init in inits:
        if time.time() > DEADLINE - 5:
            break
        # Run local search
        res, res_cost = local_search(init)
        if res_cost < best_global_cost:
            best_global_cost = res_cost
            best_global = res
    
    # If we have time, do a more aggressive SA
    if best_global and time.time() < DEADLINE - 10:
        # SA on the best solution
        cur = best_global[:]
        cur_cost = best_global_cost
        t = 10.0
        alpha = 0.99
        
        for i in range(20000):
            if time.time() > DEADLINE - 2:
                break
            t *= alpha
            if t < 0.01:
                break
                
            idx = random.randint(0, N - 1)
            new_val = random.randint(1, MAXV)
            if new_val in cur and new_val != cur[idx]:
                continue
                
            new_arr = cur[:]
            new_arr[idx] = new_val
            if len(set(new_arr)) != N:
                continue
                
            nc = cost(new_arr)
            delta = nc - cur_cost
            if delta <= 0 or random.random() < math.exp(-delta / t):
                cur = new_arr
                cur_cost = nc
                if cur_cost < best_global_cost:
                    best_global_cost = cur_cost
                    best_global = cur[:]
    
    if best_global is None:
        best_global = inits[0]
        
    return best_global
