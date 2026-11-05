import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Generate candidate numbers: 5-smooth numbers <= 10000
    # A 5-smooth number is of the form 2^a * 3^b * 5^c
    candidates = set()
    for a in range(14):  # 2^13 = 8192
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(9):  # 3^8 = 6561
            v3 = 3 ** b
            if v2 * v3 > 10000:
                break
            for c in range(7):  # 5^6 = 15625
                v5 = 5 ** c
                n = v2 * v3 * v5
                if n > 10000:
                    break
                candidates.add(n)
    
    candidates = sorted(list(candidates))
    # Ensure we have at least 12 candidates
    if len(candidates) < 12:
        # Fallback to first 12 integers
        return list(range(1, 13))
        
    # 2. Precompute prime factors for pairwise sums
    # We only need primes up to 20000 (max sum 10000+10000)
    max_sum = 20000
    # Sieve of Eratosthenes to get smallest prime factor
    spf = list(range(max_sum + 1))
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Precompute factor sets for all pairs in candidates
    # candidates index: i < j
    n_cand = len(candidates)
    pair_factors = {}
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            s = candidates[i] + candidates[j]
            pair_factors[(i, j)] = get_prime_factors(s)
            
    # 3. Objective function
    def calculate_cost(indices):
        # indices is a tuple/list of indices into candidates
        union_factors = set()
        for k in range(len(indices)):
            for m in range(k + 1, len(indices)):
                i, j = indices[k], indices[m]
                if i > j:
                    i, j = j, i
                union_factors.update(pair_factors[(i, j)])
        return len(union_factors)

    # 4. Initial Solution
    # Greedy: Start with 0,1 then add the number that minimizes the increase in distinct primes
    current_indices = [0, 1]
    current_cost = calculate_cost(current_indices)
    
    for _ in range(10):
        best_idx = -1
        best_cost = float('inf')
        for i in range(n_cand):
            if i in current_indices:
                continue
            new_indices = current_indices + [i]
            cost = calculate_cost(new_indices)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        if best_idx == -1:
            break
        current_indices.append(best_idx)
        current_cost = best_cost

    best_indices = current_indices[:]
    best_cost = current_cost
    
    # 5. Local Search / Simulated Annealing
    # We want to replace one element with another
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random index in current set to remove
        rem_pos = random.randint(0, 11)
        rem_idx = best_indices[rem_pos]
        
        # Pick a random candidate to add
        add_idx = random.randint(0, n_cand - 1)
        if add_idx in best_indices:
            continue
            
        # Create new set
        new_indices = best_indices[:]
        new_indices[rem_pos] = add_idx
        # Ensure uniqueness check is implicit in the set logic? No, best_indices is a list.
        # If add_idx is already in best_indices (at another position), we have a duplicate.
        # The problem requires DISTINCT integers.
        # So we must ensure add_idx is not in best_indices.
        if add_idx in best_indices:
            continue
            
        new_cost = calculate_cost(new_indices)
        
        # Acceptance criterion
        if new_cost < best_cost or random.random() < math.exp(-(new_cost - best_cost) / T):
            best_indices = new_indices
            best_cost = new_cost
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # 6. Final Polish: Try all single swaps again on the best found
    for _ in range(3): # A few passes
        improved = False
        for i in range(12):
            for j in range(n_cand):
                if j in best_indices:
                    continue
                new_indices = best_indices[:]
                new_indices[i] = j
                cost = calculate_cost(new_indices)
                if cost < best_cost:
                    best_indices = new_indices
                    best_cost = cost
                    improved = True
        if not improved:
            break
            
    # Convert indices back to values
    result = [candidates[i] for i in best_indices]
    
    # Verify distinctness and bounds
    if len(set(result)) != 12 or any(x > 10000 or x <= 0 for x in result):
        return list(range(1, 13))
        
    return result
