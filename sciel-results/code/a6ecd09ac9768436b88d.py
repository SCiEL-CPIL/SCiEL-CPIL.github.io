import time
import sys

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
    
    # Multiple orderings
    orderings = []
    orderings.append(sorted(other_candidates, reverse=True))
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), -x)))
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), x)))
    orderings.append(sorted(other_candidates, key=lambda x: len(pf_cache[x])))
    
    n_orderings = len(orderings)
    time_per_ordering = 12  # ~48s total for 4 orderings
    
    for i, ordering in enumerate(orderings):
        if time.time() > deadline - 3:
            break
        t_limit = min(start_time + time_per_ordering * (i + 1) + 2, deadline - 3)
        dfs(0, list(safe_primes), P_base, ordering, t_limit)
    
    # Final polish
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    for x in remaining:
        if time.time() > deadline:
            break
        if is_valid(P_best, x):
            new_P = set(P_best)
            for p in P_best:
                new_P.add(p * x)
            P_best = new_P
            best_set.append(x)
    
    return [int(x) for x in best_set]
