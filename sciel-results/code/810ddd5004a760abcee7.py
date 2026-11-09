import numpy as np
import time
import random
from fractions import Fraction

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    # 150s limit, leave 10s margin for safety
    deadline = start_time + 140.0
    
    primes = get_primes(1000)
    L_limit = 100000
    
    # Function to calculate budget using Fraction for exactness
    def calc_budget(prime_set):
        return sum(Fraction(1, p) for p in prime_set)
    
    # Generate candidate prime sets
    # We want sum(1/p) <= 1
    # Small primes are generally better for covering a prefix [1, L]
    
    sets_to_try = []
    
    # 1. Greedy smallest primes
    def greedy_set(skip_set=None, max_p=1000):
        s = []
        budget = Fraction(0)
        for p in primes:
            if p > max_p: break
            if skip_set and p in skip_set: continue
            if budget + Fraction(1, p) <= Fraction(1):
                s.append(p)
                budget += Fraction(1, p)
            else:
                # If we can't fit this prime, we might fit a larger one?
                # No, 1/p decreases. If 1/p > rem, then 1/q < rem for q>p.
                # But we want to maximize coverage. Usually filling with smallest is good.
                # However, if we skip a small prime, we can add many larger ones.
                # Let's just stop for the "greedy smallest" strategy.
                pass
        return s

    sets_to_try.append(greedy_set())
    sets_to_try.append(greedy_set(skip_set={2}))
    sets_to_try.append(greedy_set(skip_set={2, 3}))
    sets_to_try.append(greedy_set(skip_set={2, 5}))
    
    # 2. Seed-like set: primes >= 13
    s_seed = [p for p in primes if p >= 13]
    # Ensure budget <= 1
    b = calc_budget(s_seed)
    if b > Fraction(1):
        # Remove largest primes (smallest 1/p) until valid? 
        # Actually removing large primes reduces budget the least.
        # We want to keep small primes.
        # So remove from the end (largest primes).
        while s_seed and calc_budget(s_seed) > Fraction(1):
            s_seed.pop()
    sets_to_try.append(s_seed)
    
    # 3. Mixed sets: Small primes + some medium
    # Try to include 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37...
    # Greedy already does this.
    
    # Deduplicate
    unique_sets = []
    for s in sets_to_try:
        if s and not any(set(s) == set(u) for u in unique_sets):
            unique_sets.append(s)
    
    best_L = 0
    best_config = None
    
    # Evaluation function using NumPy
    # Pre-allocate covered array
    covered = np.zeros(L_limit + 1, dtype=bool)
    
    def eval_L(res_list, p_set):
        nonlocal covered
        covered[:] = False
        for i, p in enumerate(p_set):
            r = res_list[i]
            if r > 0:
                covered[r::p] = True
            else:
                # r=0 covers p, 2p, ...
                # Note: 0 is not in 1..L, but p, 2p... are.
                # covered[0] is index 0. We care about 1..L.
                # covered[0::p] sets 0, p, 2p...
                # We only check 1..L
                covered[0::p] = True
        
        # Find first False in covered[1:]
        # If all True, L = L_limit
        # np.argmax returns 0 if all False? No, if all True, ~covered is all False.
        # Let's use a different method for speed.
        # Check if covered[1:L_limit+1] has any False.
        # This is O(L).
        # To speed up, we can check chunks?
        # Or just use np.where.
        
        # Actually, for L up to 100000, a simple loop or np.argmax on ~covered is fine.
        # But we do this many times.
        
        # Optimization: Only check up to current_best_L + some margin?
        # No, we need exact L to know if we improved.
        
        # Let's stick to numpy vectorized operation
        not_covered = ~covered[1:]
        if not np.any(not_covered):
            return L_limit
        
        # Find first True in not_covered
        first_gap = np.argmax(not_covered)
        return first_gap

    # Local Search: Coordinate Descent
    def local_search(p_set, res_list, time_limit):
        k = len(p_set)
        current_L = eval_L(res_list, p_set)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle order to avoid bias
            order = list(range(k))
            random.shuffle(order)
            
            for i in order:
                p = p_set[i]
                old_r = res_list[i]
                best_r = old_r
                best_L_local = current_L
                
                # Try all residues for this prime
                # If p is large (e.g., > 50), trying all is expensive?
                # 50 evaluations * eval_L cost.
                # eval_L is O(L).
                # If p=1000, 1000 * 100000 ops = 100M ops. Too slow.
                # We should only scan all residues if p is small.
                # If p is large, use random sampling.
                
                if p < 50:
                    for r in range(p):
                        if r == old_r: continue
                        res_list[i] = r
                        L_val = eval_L(res_list, p_set)
                        if L_val > best_L_local:
                            best_L_local = L_val
                            best_r = r
                else:
                    # Random sampling for large primes
                    for _ in range(10):
                        r = random.randint(0, p - 1)
                        if r == old_r: continue
                        res_list[i] = r
                        L_val = eval_L(res_list, p_set)
                        if L_val > best_L_local:
                            best_L_local = L_val
                            best_r = r
                
                res_list[i] = best_r
                if best_r != old_r:
                    improved = True
                    if best_L_local > current_L:
                        current_L = best_L_local
        
        return current_L

    # Run local search on each set
    for idx, p_set in enumerate(unique_sets):
        if time.time() > deadline - 5:
            break
            
        k = len(p_set)
        if k == 0: continue
        
        # Multiple restarts for each set?
        # Time budget per set
        time_per_set = (deadline - time.time()) / (len(unique_sets) - idx)
        time_per_restart = max(1.0, time_per_set / 5.0) # Try up to 5 restarts
        
        for restart in range(5):
            if time.time() > deadline - 2:
                break
                
            # Initialize residues
            # Heuristic: r=0 is often good? Or random?
            # Let's try random initialization
            res_list = [random.randint(0, p - 1) for p in p_set]
            
            # Run local search
            L_val = local_search(p_set, res_list, time.time() + time_per_restart)
            
            if L_val > best_L:
                best_L = L_val
                best_config = (p_set[:], res_list[:])
                
                # If we reached the cap, we can stop
                if best_L >= L_limit:
                    break
        
        if best_L >= L_limit:
            break

    # If no config found, fallback to seed
    if best_config is None:
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        # Ensure budget
        b = calc_budget(primes_seed)
        while primes_seed and calc_budget(primes_seed) > Fraction(1):
            primes_seed.pop()
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    p_set, res_list = best_config
    return [[p, r] for p, r in zip(p_set, res_list)]
