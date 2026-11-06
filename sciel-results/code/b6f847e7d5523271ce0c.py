import time
import random

MAX_N = 511
WEIGHTS = [0.0] * (MAX_N + 1)
PRIME_FACTORS = [[] for _ in range(MAX_N + 1)]
precompute_done = False

def precompute():
    global precompute_done
    if precompute_done:
        return
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for a in range(2, MAX_N + 1):
        x = a
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        PRIME_FACTORS[a] = list(factors)
        
    for a in range(1, MAX_N + 1):
        WEIGHTS[a] = 1.0 / (512 - a)
    precompute_done = True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    precompute()

    # 1. Greedy Initialization
    candidates = list(range(1, MAX_N + 1))
    candidates.sort(key=lambda x: WEIGHTS[x], reverse=True)
    
    chosen = []
    used_primes = set()
    
    for a in candidates:
        if time.time() > deadline:
            break
        factors = PRIME_FACTORS[a]
        if all(f not in used_primes for f in factors):
            chosen.append(a)
            used_primes.update(factors)

    def get_score(S):
        return sum(WEIGHTS[a] for a in S)
    
    current_score = get_score(chosen)
    
    # Helper to find best replacements for removed numbers
    def find_best_replacements(removed_nums, current_used_primes):
        # Filter candidates: must be coprime to current_used_primes
        # We iterate through all 1..511, which is fast enough
        avail_candidates = []
        for a in range(1, MAX_N + 1):
            # Optimization: if a is in current_used_primes? No, a is a number.
            # We check if a's prime factors are free.
            if all(f not in current_used_primes for f in PRIME_FACTORS[a]):
                avail_candidates.append(a)
        
        # Sort by weight
        avail_candidates.sort(key=lambda x: WEIGHTS[x], reverse=True)
        
        best_set = []
        new_used = set()
        best_score = 0.0
        
        for a in avail_candidates:
            if all(f not in new_used for f in PRIME_FACTORS[a]):
                best_set.append(a)
                new_used.update(PRIME_FACTORS[a])
                best_score += WEIGHTS[a]
                
        return best_set, best_score

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-swaps
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            
            a = chosen[i]
            
            # Remove a
            new_chosen = chosen[:i] + chosen[i+1:]
            new_used = set()
            for x in new_chosen:
                new_used.update(PRIME_FACTORS[x])
            
            # Find best replacements
            replacements, rep_score = find_best_replacements([a], new_used)
            
            # If the best set is better, swap
            if rep_score > WEIGHTS[a] + 1e-9:
                chosen = new_chosen + replacements
                current_score += rep_score - WEIGHTS[a]
                improved = True
                break # Restart loop as chosen changed

        if improved:
            continue
            
        # Try 2-swaps if time permits
        if time.time() < deadline - 20: # Leave 20s margin
            n = len(chosen)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    
                    a1 = chosen[i]
                    a2 = chosen[j]
                    
                    new_chosen = []
                    for k, x in enumerate(chosen):
                        if k != i and k != j:
                            new_chosen.append(x)
                    
                    new_used = set()
                    for x in new_chosen:
                        new_used.update(PRIME_FACTORS[x])
                    
                    old_score = WEIGHTS[a1] + WEIGHTS[a2]
                    
                    replacements, rep_score = find_best_replacements([a1, a2], new_used)
                    
                    if rep_score > old_score + 1e-9:
                        chosen = new_chosen + replacements
                        current_score += rep_score - old_score
                        improved = True
                        break
                
                if improved:
                    break

    return chosen
