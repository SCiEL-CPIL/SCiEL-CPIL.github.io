"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_solve(primes, order):
    S = list(range(N + 1))
    k = len(S)
    # Optimization: Use a set for S? No, list is fine for N=5000.
    # But filtering list is O(N).
    
    # We need to iterate through primes in the given order.
    # However, the condition is p <= k.
    # If we process primes in an order that is not sorted, we must be careful.
    # If we process p=5 then p=3, and k drops below 3, we stop.
    # But if we process p=5 first, we check it.
    
    # To make it efficient, we can just iterate the provided order.
    # If p > k, we can break ONLY if the remaining primes are all > k.
    # Since we don't know the future primes in the order, we can't break early easily unless we sort.
    # But N=5000 is small. We can just check all primes in the order.
    
    for p in order:
        if p > k:
            # If p > k, this prime is not a constraint.
            # But subsequent primes in the list might be smaller?
            # If the order is random, yes.
            # If the order is sorted, no.
            # To be safe, we just check p <= k.
            continue
        
        # Count residues mod p
        counts = [0] * p
        for v in S:
            counts[v % p] += 1
        
        # Check if all classes are covered
        if min(counts) > 0:
            # Find the least-populated class
            min_count = min(counts)
            # To handle ties, we can pick the first one or randomize.
            # Randomizing might help escape local minima in random orders.
            candidates = [i for i, c in enumerate(counts) if c == min_count]
            target_class = random.choice(candidates)
            
            # Remove all elements in that class
            S = [v for v in S if v % p != target_class]
            k = len(S)
            
    return S

def construct():
    primes = _primes(N)
    best_S = []
    
    # Strategy 1: Increasing order (Seed behavior)
    S_inc = _greedy_solve(primes, primes)
    if len(S_inc) > len(best_S):
        best_S = S_inc
        
    # Strategy 2: Decreasing order
    S_dec = _greedy_solve(primes, primes[::-1])
    if len(S_dec) > len(best_S):
        best_S = S_dec
        
    # Strategy 3: Random orders
    # Time limit check
    start_time = time.time()
    max_time = 290 # Leave 10s buffer
    
    while time.time() - start_time < max_time:
        random_order = primes[:]
        random.shuffle(random_order)
        S_rand = _greedy_solve(primes, random_order)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            # Optional: if we find a very good one, we can stop? 
            # But let's keep searching.
            
    return best_S
# EVOLVE-BLOCK-END
