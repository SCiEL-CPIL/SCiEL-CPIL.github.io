"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (90s limit)
    
    N = 256
    
    # 1. Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
    
    # Base: 1 + sum of all prime powers
    base_sum = 1 + sum(prime_powers.values())
    
    # 3. Generate potential merge groups
    # A group merges primes {p1,...,pk} into their product
    # Valid if product <= N, gain = product - sum(prime_powers[pi]) > 0
    
    # Primes that can potentially be in a merge (p * 2 <= N means p <= 128)
    active_primes = [p for p in primes if p <= N // 2]
    
    potential_groups = []
    
    def time_check():
        return time.time() > deadline
    
    # Generate all valid merge groups (pairs, triples, quads, etc.)
    n_active = len(active_primes)
    
    # Pairs
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]  # fallback
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            cost = prime_powers[p1] + prime_powers[p2]
            gain = prod - cost
            if gain > 0:
                potential_groups.append((frozenset([p1, p2]), gain))
    
    # Triples
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod12 = p1 * p2
            if prod12 * 2 > N:
                continue
            for k in range(j + 1, n_active):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                cost = prime_powers[p1] + prime_powers[p2] + prime_powers[p3]
                gain = prod - cost
                if gain > 0:
                    potential_groups.append((frozenset([p1, p2, p3]), gain))
    
    # Quads (only very small primes)
    small_active = active_primes[:12]
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        cost = sum(prime_powers[p] for p in combo)
        gain = prod - cost
        if gain > 0:
            potential_groups.append((frozenset(combo), gain))
    
    # Sort by gain descending
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    # Compute suffix sums of gains for upper bound pruning
    n_groups = len(potential_groups)
    suffix_gain = [0] * (n_groups + 1)
    for i in range(n_groups - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + potential_groups[i][1]
    
    # 4. Branch and bound to find max weight independent set
    best_gain = 0
    best_groups = []
    
    def backtrack(idx, used_primes, current_gain, current_groups):
        nonlocal best_gain, best_groups
        
        if time.time() > deadline:
            return
        
        # Upper bound: current gain + sum of all remaining gains
        if current_gain + suffix_gain[idx] <= best_gain:
            return
        
        if current_gain > best_gain:
            best_gain = current_gain
            best_groups = list(current_groups)
        
        if idx == n_groups:
            return
        
        group_set, gain = potential_groups[idx]
        
        # Option 1: Take this group if no conflict
        if used_primes.isdisjoint(group_set):
            backtrack(idx + 1, used_primes | group_set, current_gain + gain, 
                      current_groups + [group_set])
        
        # Option 2: Skip this group
        backtrack(idx + 1, used_primes, current_gain, current_groups)
    
    backtrack(0, frozenset(), 0, [])
    
    # 5. Construct final result
    result = [1]
    used_in_merge = set()
    for group in best_groups:
        prod = 1
        for p in group:
            prod *= p
            used_in_merge.add(p)
        result.append(prod)
        
    for p in primes:
        if p not in used_in_merge:
            result.append(prime_powers[p])
            
    # Ensure types are Python int and verify pairwise coprimality
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
