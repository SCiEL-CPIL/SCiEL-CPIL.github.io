import time
import sys
from itertools import combinations

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
    
    # Precompute prime factorizations
    pf_cache = {x: _prime_factors(x) for x in range(2, 41)}
    
    # Candidate base sets: large primes and prime powers that are "independent"
    # The 4 largest primes <= 40: 37, 31, 29, 23
    # Also consider: 32=2^5, 27=3^3, 16=2^4, 8=2^3, 9=3^2, 4=2^2
    # Prime powers of distinct primes are independent
    
    base_sets = [
        [23, 29, 31, 37],
        [23, 29, 31, 32],
        [23, 29, 31, 27],
        [23, 29, 37, 32],
        [29, 31, 37, 32],
        [23, 31, 37, 32],
        [23, 29, 31, 27],
        [23, 29, 37, 27],
        [29, 31, 37, 27],
        [23, 29, 31, 8],
        [23, 29, 31, 9],
        [23, 29, 31, 4],
        [23, 29, 31, 2],
        [23, 29, 31, 3],
        [23, 29, 31, 5],
        [23, 29, 31, 7],
        [23, 29, 31, 11],
        [23, 29, 31, 13],
        [23, 29, 31, 17],
        [23, 29, 31, 19],
    ]
    
    best_set = [23, 29, 31, 37]
    best_size = 4
    
    def compute_products(S):
        P = {1}
        for x in S:
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            P = new_P
        return P
    
    def is_valid(P, x):
        for p in P:
            if p * x in P:
                return False
        return True
    
    def dfs(index, S, P, candidates, time_limit, current_best_size):
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
            dfs(index + 1, S + [x], new_P, candidates, time_limit, best_size)
        
        dfs(index + 1, S, P, candidates, time_limit, best_size)
    
    # Try each base set with time-limited DFS
    n_bases = len(base_sets)
    time_per_base = 20  # ~40s total for 2 bases, but we'll do more with less time each
    
    # Sort base sets by heuristic: prefer those with more "spread" in prime factors
    for bi, base in enumerate(base_sets):
        if time.time() > deadline - 5:
            break
        
        base_set = set(base)
        remaining = [x for x in range(2, 41) if x not in base_set]
        
        # Order remaining candidates: those with more unique prime factors first, then larger
        def order_key(x):
            pf = pf_cache[x]
            # Score: number of primes that appear only in this candidate among remaining+base
            unique_count = 0
            for p in pf:
                only_in_x = True
                for y in remaining:
                    if y != x and p in pf_cache[y]:
                        only_in_x = False
                        break
                if only_in_x:
                    unique_count += 1
            return (-unique_count, -x)
        
        ordered = sorted(remaining, key=order_key)
        
        P_base = compute_products(base)
        
        # Time limit for this base set
        t_limit = min(start_time + (bi + 1) * (40 / n_bases), deadline - 5)
        
        dfs(0, list(base), P_base, ordered, t_limit, best_size)
    
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
