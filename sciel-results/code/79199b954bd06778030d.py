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
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Precompute prime factorizations
    pf_cache = {x: _prime_factors(x) for x in range(2, 41)}
    
    # Safe primes > 20: their multiples in [2,40] are only themselves
    # So any product containing one of these is uniquely identifiable
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]
    
    best_set = list(safe_primes)
    best_size = len(best_set)
    
    def compute_products(S):
        P = {1}
        for x in S:
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            P = new_P
        return P
    
    P_base = compute_products(safe_primes)
    
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
        
        # Pruning: can't beat best
        if len(S) + (len(candidates) - index) <= best_size:
            return
        
        x = candidates[index]
        
        # Try including x first (build larger sets quickly)
        if is_valid(P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs(index + 1, S + [x], new_P, candidates, time_limit)
        
        # Try excluding x
        dfs(index + 1, S, P, candidates, time_limit)
    
    # Create multiple orderings to try
    # Ordering 1: fewer prime factors first, then larger values
    ordering1 = sorted(other_candidates, key=lambda x: (len(pf_cache[x]), -x))
    # Ordering 2: larger values first
    ordering2 = sorted(other_candidates, reverse=True)
    # Ordering 3: fewer prime factors, then smaller values
    ordering3 = sorted(other_candidates, key=lambda x: (len(pf_cache[x]), x))
    # Ordering 4: primes first (1 factor), then semi-primes, etc., within each group larger first
    ordering4 = sorted(other_candidates, key=lambda x: (len(pf_cache[x]), -x))
    
    orderings = [ordering1, ordering2, ordering3]
    
    # Allocate time: give most time to the best ordering, less to others
    # First ordering gets ~40s, remaining share the rest
    t_limits = []
    remaining_time = deadline - 3
    for i in range(len(orderings)):
        if i == 0:
            t_limits.append(min(start_time + 35, deadline - 3))
        else:
            t_limits.append(min(start_time + 35 + (i - 1) * 5, deadline - 3))
    
    for i, ordering in enumerate(orderings):
        if time.time() > t_limits[i]:
            break
        dfs(0, list(safe_primes), P_base, ordering, t_limits[i])
    
    # Final polish: greedily add any remaining valid numbers
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = compute_products(best_set)
    
    changed = True
    while changed:
        changed = False
        for x in remaining:
            if time.time() > deadline:
                break
            if is_valid(P_best, x):
                new_P = set(P_best)
                for p in P_best:
                    new_P.add(p * x)
                P_best = new_P
                best_set.append(x)
                best_set_set.add(x)
                remaining.remove(x)
                changed = True
        if time.time() > deadline:
            break
    
    return [int(x) for x in best_set]
