import time
from math import gcd

def construct():
    deadline = time.time() + 55.0
    start = time.time()
    
    # --- Precomputation ---
    # multiples_of[k] = list of m in [2,256] such that m % k == 0
    multiples_of = [[] for _ in range(257)]
    for k in range(1, 257):
        for m in range(k, 257, k):
            if m >= 2:
                multiples_of[k].append(m)
                
    # Precompute prime factorization info
    lpf = [0] * 257  # largest prime factor
    num_prime_factors = [0] * 257  # count of distinct prime factors
    
    for i in range(2, 257):
        if lpf[i] == 0: # i is prime
            for j in range(i, 257, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    # Count distinct prime factors
    for i in range(2, 257):
        n = i
        while n > 1:
            p = lpf[n]
            num_prime_factors[i] += 1
            while n % p == 0:
                n //= p
    
    candidates = list(range(2, 257))
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]

    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set (boolean array/list).
        Returns True if valid, False otherwise.
        """
        for b in range(2, 257):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check 1: a | b*c => (a/g) | c
            k1 = a // g
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check 2: b | a*c => (b/g) | c
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
                    
        return True

    def greedy_construction(order):
        in_set = [False] * 257
        current_set = []
        for a in order:
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    def local_search_improve(S):
        """
        Try to improve set S by removing one element and adding one or more.
        Returns improved set if found, else S.
        """
        S_set = set(S)
        in_set = [False] * 257
        for x in S:
            in_set[x] = True
        
        best_size = len(S)
        best_S = S[:]
        
        # Try removing each element x and then greedily adding from remaining
        for x in S:
            # Temporarily remove x
            in_set[x] = False
            S_temp = [y for y in S if y != x]
            S_temp_set = S_set - {x}
            
            # Try to add elements from candidates not in S_temp
            added_any = False
            for y in candidates:
                if y in S_temp_set:
                    continue
                if can_add(y, in_set):
                    in_set[y] = True
                    S_temp.append(y)
                    S_temp_set.add(y)
                    added_any = True
            
            if len(S_temp) > best_size:
                best_size = len(S_temp)
                best_S = S_temp[:]
            
            # Restore
            in_set[x] = True
        
        return best_S

    best_set = []
    
    # --- Strategy 1: Deterministic Greedy with specific orderings ---
    # Order 1: Primes first (descending), then composites by fewer prime factors, then descending value
    order_primes_first = sorted(primes, reverse=True) + sorted(composites, key=lambda x: (num_prime_factors[x], -x))
    S = greedy_construction(order_primes_first)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 2: Descending Value
    order_desc = sorted(candidates, reverse=True)
    S = greedy_construction(order_desc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 3: Fewest prime factors first, then descending value
    order_fewest_pf = sorted(candidates, key=lambda x: (num_prime_factors[x], -x))
    S = greedy_construction(order_fewest_pf)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 4: Primes descending, then composites descending
    order_primes_composites_desc = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_construction(order_primes_composites_desc)
    if len(S) > len(best_set):
        best_set = S[:]

    # Apply local search to best deterministic result
    if len(best_set) > 0:
        improved = local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # --- Strategy 2: Randomized Greedy Restarts with bias ---
    import random
    
    # Weighted random order: primes have higher weight, fewer prime factors have higher weight
    def weighted_random_order():
        weights = []
        for x in candidates:
            w = 1.0
            if lpf[x] == x:  # prime
                w *= 3.0
            # Fewer prime factors = safer
            w *= (4.0 - num_prime_factors[x])
            weights.append(w)
        
        # Sample without replacement using weights
        order = []
        remaining = list(range(len(candidates)))
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = random.random() * total_w
            cum = 0
            for i in remaining:
                cum += weights[i]
                if r <= cum:
                    order.append(candidates[i])
                    remaining.remove(i)
                    break
        return order

    num_restarts = 0
    max_restarts = 5000
    
    while time.time() < deadline - 5.0 and num_restarts < max_restarts:
        # Mix of pure random and weighted random
        if num_restarts % 3 == 0:
            order = weighted_random_order()
        else:
            order = candidates[:]
            random.shuffle(order)
        
        S = greedy_construction(order)
        num_restarts += 1
        
        if len(S) > len(best_set):
            best_set = S[:]
            # Try local search on new best
            improved = local_search_improve(best_set)
            if len(improved) > len(best_set):
                best_set = improved

    # Final local search on best_set
    if len(best_set) > 0:
        improved = local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    result = [int(x) for x in best_set]
    return result
