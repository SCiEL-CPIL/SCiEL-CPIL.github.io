import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    MAX_VAL = 10000
    N = 12
    
    # Precompute smallest prime factor up to 2*MAX_VAL
    MAX_SUM = 2 * MAX_VAL
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n == 1:
            return frozenset()
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return frozenset(factors)
    
    # Precompute prime factors for all possible sums (2 to 2*MAX_VAL)
    # This is a big list but fast to build and very fast to query
    pf_cache = [None] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        pf_cache[s] = get_prime_factors(s)
    
    def evaluate(a):
        """Return the set of all prime factors of pairwise sums."""
        primes = set()
        for i in range(N):
            for j in range(i+1, N):
                primes.update(pf_cache[a[i] + a[j]])
        return primes
    
    best_primes = None
    best_set = None
    
    def update_best(a, primes):
        nonlocal best_primes, best_set
        if best_primes is None or len(primes) < len(best_primes):
            best_primes = primes
            best_set = a[:]
    
    # --- Strategy 1: Base multiplier with local search ---
    # x_i = base * k_i. Sums = base*(k_i + k_j).
    # Prime factors of sum = factors(base) U factors(k_i+k_j).
    # We search over k_i in a restricted range.
    
    bases = [30, 60, 120, 210, 360, 840, 2, 4, 6, 8, 10, 12, 15, 18, 20, 24, 25, 30,
             720, 1260, 1680, 2520, 5040, 10080, 420, 840, 1260, 1680, 2520,
             3, 5, 7, 9, 11, 13, 14, 16, 21, 22, 26, 27, 28, 32, 33, 34, 35, 36]
    # Deduplicate while preserving order
    seen_bases = set()
    unique_bases = []
    for b in bases:
        if b not in seen_bases:
            seen_bases.add(b)
            unique_bases.append(b)
    bases = unique_bases
    
    for base in bases:
        if time.time() > deadline - 5.0:
            break
        
        base_factors = pf_cache[base]
        max_k = MAX_VAL // base
        
        if max_k < N:
            continue
        
        # Search limit for k values
        search_limit = min(max_k, 150)
        
        # Initial seed: 1, 2, ..., N
        cur_k = list(range(1, N + 1))
        
        def get_score_k(k_list):
            union = set(base_factors)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    union.update(pf_cache[k_list[i] + k_list[j]])
            return union
        
        cur_primes = get_score_k(cur_k)
        
        # Local search with hill climbing
        improved = True
        iterations = 0
        while improved and iterations < 300 and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5.0:
                    break
                
                old_k = cur_k[i]
                # Try replacing with values from 1..search_limit
                for new_k in range(1, search_limit + 1):
                    if new_k == old_k or new_k in cur_k:
                        continue
                    
                    cur_k[i] = new_k
                    new_primes = get_score_k(cur_k)
                    
                    if len(new_primes) < len(cur_primes):
                        cur_primes = new_primes
                        improved = True
                        break
                    cur_k[i] = old_k
                
                if improved:
                    # Convert to x values and update best
                    x_vals = [base * k for k in cur_k]
                    if all(x <= MAX_VAL for x in x_vals):
                        update_best(x_vals, cur_primes)
                    break  # Restart i-loop
        
        # Also try a few random restarts for this base
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            cur_k = random.sample(range(1, search_limit + 1), N)
            cur_primes = get_score_k(cur_k)
            
            improved = True
            iterations = 0
            while improved and iterations < 100 and time.time() < deadline - 5.0:
                improved = False
                iterations += 1
                indices = list(range(N))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 5.0:
                        break
                    old_k = cur_k[i]
                    for new_k in range(1, search_limit + 1):
                        if new_k == old_k or new_k in cur_k:
                            continue
                        cur_k[i] = new_k
                        new_primes = get_score_k(cur_k)
                        if len(new_primes) < len(cur_primes):
                            cur_primes = new_primes
                            improved = True
                            break
                        cur_k[i] = old_k
                    if improved:
                        x_vals = [base * k for k in cur_k]
                        if all(x <= MAX_VAL for x in x_vals):
                            update_best(x_vals, cur_primes)
                        break
        
        if best_primes is not None and len(best_primes) <= 4:
            break
    
    # --- Strategy 2: Direct search with smooth numbers ---
    if best_primes is None or len(best_primes) > 5:
        # Build pool of 5-smooth numbers
        pool = set()
        for a in range(0, 15):
            for b in range(0, 15):
                for c in range(0, 15):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= val <= MAX_VAL:
                        pool.add(val)
        pool = sorted(pool)
        
        if len(pool) >= N:
            # Try a few random subsets
            for restart in range(50):
                if time.time() > deadline - 5.0:
                    break
                cur_x = random.sample(pool[:200], N)
                cur_primes = evaluate(cur_x)
                
                improved = True
                iterations = 0
                while improved and iterations < 200 and time.time() < deadline - 5.0:
                    improved = False
                    iterations += 1
                    indices = list(range(N))
                    random.shuffle(indices)
                    
                    for i in indices:
                        if time.time() > deadline - 5.0:
                            break
                        old_x = cur_x[i]
                        # Try nearby smooth numbers
                        idx = pool.index(old_x)
                        for next_idx in range(max(0, idx - 10), min(len(pool), idx + 50)):
                            new_x = pool[next_idx]
                            if new_x == old_x or new_x in cur_x:
                                continue
                            cur_x[i] = new_x
                            new_primes = evaluate(cur_x)
                            if len(new_primes) < len(cur_primes):
                                cur_primes = new_primes
                                improved = True
                                break
                            cur_x[i] = old_x
                        if improved:
                            update_best(cur_x, cur_primes)
                            break
    
    # --- Strategy 3: General random restarts with local search ---
    for restart in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Initialize with random values
        a = random.sample(range(1, MAX_VAL + 1), N)
        current_primes = evaluate(a)
        update_best(a, current_primes)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                old_val = a[i]
                # Try a few random candidates
                for _ in range(15):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in a:
                        continue
                    
                    a[i] = new_val
                    new_primes = evaluate(a)
                    
                    if len(new_primes) < len(current_primes):
                        current_primes = new_primes
                        improved = True
                        update_best(a, current_primes)
                        break
                    a[i] = old_val
                
                if improved:
                    break
    
    # --- Final polish: try to further reduce prime count on best solution ---
    if best_set is not None and best_primes is not None:
        cur_x = best_set[:]
        cur_primes = best_primes
        polish_deadline = deadline - 1.0
        
        improved = True
        polish_iter = 0
        while improved and polish_iter < 200 and time.time() < polish_deadline:
            improved = False
            polish_iter += 1
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > polish_deadline:
                    break
                
                old_val = cur_x[i]
                # Try a wider range of candidate replacements
                candidates = []
                # Nearby values
                for delta in range(-50, 51):
                    nv = old_val + delta
                    if 1 <= nv <= MAX_VAL and nv != old_val and nv not in cur_x:
                        candidates.append(nv)
                # Random candidates
                for _ in range(30):
                    nv = random.randint(1, MAX_VAL)
                    if nv != old_val and nv not in cur_x:
                        candidates.append(nv)
                
                for new_val in candidates:
                    if time.time() > polish_deadline:
                        break
                    cur_x[i] = new_val
                    new_primes = evaluate(cur_x)
                    if len(new_primes) < len(cur_primes):
                        cur_primes = new_primes
                        improved = True
                        break
                    cur_x[i] = old_val
                
                if improved:
                    best_set = cur_x[:]
                    best_primes = cur_primes
                    break
    
    if best_set:
        return best_set
    
    # Fallback
    return list(range(1, N+1))
