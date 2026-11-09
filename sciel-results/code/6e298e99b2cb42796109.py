import time
import sys
import random

def _prime_factors(n):
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

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]
    
    best_set = list(safe_primes)
    best_size = len(best_set)
    
    P_base = {1}
    for x in safe_primes:
        new_P = set(P_base)
        for p in P_base:
            new_P.add(p * x)
        P_base = new_P
    
    pf_cache = {x: _prime_factors(x) for x in range(2, 41)}
    
    def is_valid(P, x):
        for p in P:
            if p * x in P:
                return False
        return True
    
    def dfs(index, S, P, candidates, time_limit):
        nonlocal best_set, best_size
        if time.time() > time_limit:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        if index == len(candidates):
            return
        
        if len(S) + (len(candidates) - index) <= best_size:
            return
        
        x = candidates[index]
        
        if is_valid(P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs(index + 1, S + [x], new_P, candidates, time_limit)
        
        dfs(index + 1, S, P, candidates, time_limit)
    
    # Generate more diverse orderings
    orderings = []
    # Original orderings
    orderings.append(sorted(other_candidates, reverse=True))
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), -x)))
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), x)))
    orderings.append(sorted(other_candidates, key=lambda x: len(pf_cache[x])))
    
    # Add orderings based on "conflict potential" - numbers with more prime factors may conflict more
    # Also try ascending order
    orderings.append(sorted(other_candidates))
    # Try ordering by number of distinct prime factors, then by value
    orderings.append(sorted(other_candidates, key=lambda x: (-len(pf_cache[x]), x)))
    # Try ordering by number of distinct prime factors, then by value descending
    orderings.append(sorted(other_candidates, key=lambda x: (-len(pf_cache[x]), -x)))
    
    # Add some randomized orderings for diversity
    random.seed(42)
    for _ in range(3):
        rand_order = other_candidates.copy()
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    n_orderings = len(orderings)
    # Allocate time: give more time to structured orderings, less to random
    # 7 structured + 3 random = 10 total
    # Give 5s to each structured (35s) and 5s to each random (15s) = 50s total
    time_per_structured = 5
    time_per_random = 5
    
    for i, ordering in enumerate(orderings):
        if time.time() > deadline - 3:
            break
        is_random = i >= 7  # indices 7,8,9 are random
        t_per = time_per_random if is_random else time_per_structured
        t_limit = min(start_time + t_per * (i + 1) + 2, deadline - 3)
        dfs(0, list(safe_primes), P_base, ordering, t_limit)
    
    # Final polish: try to add remaining elements
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    # Try adding elements in different orders
    for order in [sorted(remaining), sorted(remaining, reverse=True)]:
        P_temp = set(P_best)
        S_temp = list(best_set)
        for x in order:
            if time.time() > deadline:
                break
            if is_valid(P_temp, x):
                new_P = set(P_temp)
                for p in P_temp:
                    new_P.add(p * x)
                P_temp = new_P
                S_temp.append(x)
        
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_set = S_temp
            P_best = P_temp
    
    # Try swapping: for each element in best_set, try replacing it with a remaining element
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    for i, elem in enumerate(list(best_set)):
        if time.time() > deadline:
            break
        for new_elem in remaining:
            if time.time() > deadline:
                break
            # Try replacing elem with new_elem
            new_S = list(best_set)
            new_S[i] = new_elem
            new_P = {1}
            valid = True
            for x in new_S:
                new_P_temp = set(new_P)
                for p in new_P:
                    new_P_temp.add(p * x)
                new_P = new_P_temp
            
            # Check if all subset products are distinct
            if len(new_P) == 2 ** len(new_S):
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_set = new_S
                    P_best = new_P
    
    return [int(x) for x in best_set]
