import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    primes = _primes(N)
    num_primes = len(primes)
    
    # Precompute indices for each prime and residue?
    # Or just generate on the fly?
    # Generating on the fly: range(residue, N+1, p)
    
    # State: chosen residue for each prime
    # r[i] is the residue for primes[i]
    r = [random.randint(0, p - 1) for p in primes]
    
    # cover_count[x] = how many primes currently cover x
    cover_count = np.zeros(N + 1, dtype=np.int32)
    
    def update_cover(p_idx, old_res, new_res):
        """Update cover_count and return delta_cost (change in number of covered elements)"""
        p = primes[p_idx]
        delta = 0
        
        # Remove old residue class
        for x in range(old_res, N + 1, p):
            cover_count[x] -= 1
            if cover_count[x] == 0:
                delta -= 1
        
        # Add new residue class
        for x in range(new_res, N + 1, p):
            cover_count[x] += 1
            if cover_count[x] == 1:
                delta += 1
                
        return delta

    # Initial cost calculation
    current_cost = 0
    for i, p in enumerate(primes):
        for x in range(r[i], N + 1, p):
            cover_count[x] += 1
    current_cost = np.sum(cover_count > 0)
    
    best_cost = current_cost
    best_r = r.copy()
    best_S = None
    
    # Helper to construct S from r
    def get_S(r_config):
        # Re-create cover count for this config to be safe, or reuse if we track it?
        # For simplicity and correctness, recompute from scratch for best candidate
        cc = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            for x in range(r_config[i], N + 1, p):
                cc[x] += 1
        # S is the set of x where cc[x] == 0
        S = [x for x in range(N + 1) if cc[x] == 0]
        return S

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    iterations = 0
    while time.time() < deadline:
        if T < T_min:
            # Restart or break
            T = 1.0 # Restart SA
            # Or break if we want to be conservative
            # Let's restart with a random perturbation of best solution
            r = best_r.copy()
            # Perturb a few random primes
            for _ in range(10):
                idx = random.randint(0, num_primes - 1)
                p = primes[idx]
                r[idx] = random.randint(0, p - 1)
            # Recalculate cover_count and cost from scratch for this new start
            cover_count = np.zeros(N + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                for x in range(r[i], N + 1, p):
                    cover_count[x] += 1
            current_cost = np.sum(cover_count > 0)
            if current_cost < best_cost:
                best_cost = current_cost
                best_r = r.copy()
        
        # Pick a random prime to change
        p_idx = random.randint(0, num_primes - 1)
        p = primes[p_idx]
        
        old_res = r[p_idx]
        new_res = random.randint(0, p - 1)
        
        if new_res == old_res:
            continue
            
        delta = update_cover(p_idx, old_res, new_res)
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept move
            r[p_idx] = new_res
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_r = r.copy()
                
        T *= cooling_rate
        iterations += 1
        
        # Update best_S periodically to ensure we have a valid return value
        if iterations % 1000 == 0:
            best_S = get_S(best_r)
            
    # Final calculation of best S
    best_S = get_S(best_r)
    
    return best_S
