import time
from math import gcd

# Precompute prime factors for 1..511
def get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precomputation
MAX_N = 511
PRIME_FACTORS = [None] * (MAX_N + 1)
for i in range(1, MAX_N + 1):
    PRIME_FACTORS[i] = get_prime_factors(i)

# Weights
WEIGHTS = [0.0] * (MAX_N + 1)
for i in range(1, MAX_N + 1):
    WEIGHTS[i] = 1.0 / (512.0 - i)

def construct():
    start_time = time.time()
    TIMEOUT = 110.0  # Safety margin
    
    # 1. Greedy Initialization
    chosen = []
    used_primes = set()
    
    # Sort candidates by weight descending (which is value descending)
    candidates = list(range(511, 0, -1))
    
    for a in candidates:
        if time.time() - start_time > TIMEOUT:
            break
        if not (PRIME_FACTORS[a] & used_primes):
            chosen.append(a)
            used_primes.update(PRIME_FACTORS[a])
            
    current_score = sum(WEIGHTS[a] for a in chosen)
    best_score = current_score
    best_chosen = list(chosen)
    
    # 2. Local Search (1-for-1 swaps and fills)
    improved = True
    while improved:
        if time.time() - start_time > TIMEOUT:
            break
        improved = False
        
        # Try 1-for-1 swaps
        # We iterate through current chosen items
        # To be efficient, we can try to replace each 'a' with the best possible 'b'
        for i in range(len(chosen)):
            if time.time() - start_time > TIMEOUT:
                break
            a = chosen[i]
            # Temporarily remove a's primes
            temp_used = used_primes - PRIME_FACTORS[a]
            
            best_b = -1
            best_gain = 0.0
            
            # Find best b not in chosen
            # We can iterate all 1..511
            for b in range(511, 0, -1):
                if b in chosen:
                    continue
                if not (PRIME_FACTORS[b] & temp_used):
                    gain = WEIGHTS[b] - WEIGHTS[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b != -1:
                # Perform swap
                chosen[i] = best_b
                used_primes = temp_used | PRIME_FACTORS[best_b]
                current_score += best_gain
                improved = True
                # Update best if better
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(chosen)
                
        # Try to add new numbers (Fill)
        # Re-sort candidates? Or just scan?
        # Since we want max weight, scan 511 down to 1
        added = True
        while added:
            if time.time() - start_time > TIMEOUT:
                break
            added = False
            for b in range(511, 0, -1):
                if b in chosen:
                    continue
                if not (PRIME_FACTORS[b] & used_primes):
                    chosen.append(b)
                    used_primes.update(PRIME_FACTORS[b])
                    current_score += WEIGHTS[b]
                    added = True
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(chosen)
                    break # Restart scan to pick next best available

    return best_chosen
