import time
from math import gcd

# Precompute prime factors for 1..511
def get_prime_factors(n):
    factors = set()
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

# Global precomputation
MAX_N = 512
PRIME_FACTORS = [set() for _ in range(MAX_N)]
WEIGHTS = [0.0] * MAX_N

for i in range(1, MAX_N):
    PRIME_FACTORS[i] = get_prime_factors(i)
    WEIGHTS[i] = 1.0 / (MAX_N - i)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Greedy Initialization
    chosen = []
    chosen_set = set()
    used_primes = set()
    score = 0.0
    
    for a in range(MAX_N - 1, 0, -1):
        if time.time() > deadline:
            break
        if PRIME_FACTORS[a].isdisjoint(used_primes):
            chosen.append(a)
            chosen_set.add(a)
            used_primes.update(PRIME_FACTORS[a])
            score += WEIGHTS[a]
            
    if not chosen:
        return [1] # Fallback, though 1 is always coprime to everything

    # 2. Local Search (1-Opt)
    # We try to swap any element in 'chosen' with any element not in 'chosen'
    # to improve the score.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over a copy of chosen to allow modification
        for i, a in enumerate(list(chosen)):
            if time.time() > deadline:
                return chosen
            
            # Temporarily remove 'a'
            used_primes.difference_update(PRIME_FACTORS[a])
            score -= WEIGHTS[a]
            chosen_set.remove(a)
            # Remove from list (we'll re-insert or replace)
            # We'll just track the best replacement for this slot
            
            best_b = -1
            best_gain = 0.0
            
            # Try to find best 'b' to replace 'a'
            # Iterate b from high to low to prioritize high weights
            for b in range(MAX_N - 1, 0, -1):
                if b in chosen_set:
                    continue
                if PRIME_FACTORS[b].isdisjoint(used_primes):
                    gain = WEIGHTS[b]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            # If we found a better b, swap
            if best_b != -1:
                # Perform swap
                chosen[i] = best_b
                chosen_set.add(best_b)
                used_primes.update(PRIME_FACTORS[best_b])
                score += best_gain
                improved = True
                # Restart the loop to check for further improvements
                break 
            else:
                # No improvement for this 'a', restore 'a'
                chosen_set.add(a)
                used_primes.update(PRIME_FACTORS[a])
                score += WEIGHTS[a]

    return chosen
