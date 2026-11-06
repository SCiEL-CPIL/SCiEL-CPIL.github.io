import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Precomputation
    MAX_SUM = 20000
    # Sieve of Eratosthenes to find smallest prime factor
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Map primes to indices
    primes = set()
    prime_to_idx = {}
    for i in range(2, MAX_SUM + 1):
        p = spf[i]
        if p not in prime_to_idx:
            prime_to_idx[p] = len(prime_to_idx)
            primes.add(p)
            
    num_primes = len(prime_to_idx)
    
    # Precompute factor sets for each number up to MAX_SUM
    # factor_sets[x] = list of prime indices dividing x
    factor_sets = [None] * (MAX_SUM + 1)
    factor_sets[0] = []
    factor_sets[1] = []
    for x in range(2, MAX_SUM + 1):
        n = x
        fs = set()
        while n > 1:
            p = spf[n]
            fs.add(prime_to_idx[p])
            while n % p == 0:
                n //= p
        factor_sets[x] = fs

    def get_cost(arr):
        """Calculate the number of distinct prime factors in all pairwise sums."""
        counts = [0] * num_primes
        n = len(arr)
        for i in range(n):
            ai = arr[i]
            for j in range(i + 1, n):
                aj = arr[j]
                s = ai + aj
                for p_idx in factor_sets[s]:
                    counts[p_idx] += 1
        return sum(1 for c in counts if c > 0)

    def local_search(arr, max_time):
        """Hill climbing local search."""
        best_arr = list(arr)
        best_cost = get_cost(best_arr)
        n = len(best_arr)
        
        # Precompute initial counts for fast delta evaluation?
        # Actually, recalculating get_cost is O(N^2 * avg_factors). N=12.
        # 12*11/2 = 66 sums.
        # Each sum has ~2-4 factors.
        # 66 * 4 = 264 operations.
        # Very fast. We can just recalculate.
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try replacing each element
            for i in range(n):
                old_val = best_arr[i]
                # Try candidates
                # Strategy: Try small numbers first, then random
                # To avoid duplicates, ensure v not in best_arr (except at i)
                
                # Generate candidates
                # 1. Neighbors
                # 2. Random small
                # 3. Random large
                
                # Let's try a set of candidates
                candidates = set()
                # Small range
                for v in range(1, 50):
                    candidates.add(v)
                # Random
                for _ in range(20):
                    candidates.add(random.randint(1, 10000))
                
                for v in candidates:
                    if v == old_val:
                        continue
                    if v in best_arr:
                        continue
                    
                    new_arr = list(best_arr)
                    new_arr[i] = v
                    new_cost = get_cost(new_arr)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_arr = new_arr
                        improved = True
                        break # Restart search from new state
                if improved:
                    break
        return best_arr, best_cost

    best_overall_arr = list(range(1, 13))
    best_overall_cost = get_cost(best_overall_arr)
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Generate initial set
        if restarts == 1:
            init_arr = list(range(1, 13))
        else:
            # Random set of 12 distinct numbers from 1..100
            init_arr = random.sample(range(1, 101), 12)
            # Or mix
            if restarts % 2 == 0:
                init_arr = random.sample(range(1, 501), 12)
        
        arr, cost = local_search(init_arr, deadline)
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_arr = arr
            # If we found a very good solution, maybe stop early?
            # Theoretical minimum is likely 4 or 5?
            if best_overall_cost <= 4:
                break

    return best_overall_arr

# EVOLVE-BLOCK-END
