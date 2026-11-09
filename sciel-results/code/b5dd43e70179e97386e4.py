"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
_pf_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    for i in range(2, MAX_SUM + 1):
        if not _pf_cache[i]:
            n = i
            f = 2
            while f * f <= n:
                if n % f == 0:
                    _pf_cache[i].add(f)
                    while n % f == 0:
                        n //= f
                f += 1
            if n > 1:
                _pf_cache[i].add(n)

_init_prime_factors()

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

PRIMES = get_primes(100)

def construct():
    n = 12
    max_val = 10000
    
    # Try prime sets of increasing size
    # Start with 5 primes, go up to 25
    for k in range(5, 26):
        P_set = set(PRIMES[:k])
        
        # Precompute is_smooth for this P
        is_smooth = [False] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            factors = _pf_cache[s]
            if all(f in P_set for f in factors):
                is_smooth[s] = True
        
        # DFS to find 12 numbers
        # Use a node limit to prevent hanging
        node_limit = 200000
        nodes_visited = [0]
        
        def dfs(current_set, candidates):
            if nodes_visited[0] > node_limit:
                return None
            
            if len(current_set) == n:
                return current_set
            
            # Heuristic: Try candidates in order
            for idx in range(len(candidates)):
                nodes_visited[0] += 1
                if nodes_visited[0] > node_limit:
                    return None
                
                x = candidates[idx]
                
                # Filter remaining candidates
                new_candidates = []
                # Optimization: local variable for is_smooth
                is_sm = is_smooth
                for y in candidates[idx+1:]:
                    if is_sm[x + y]:
                        new_candidates.append(y)
                
                # Pruning: if not enough candidates left
                if len(new_candidates) < n - (len(current_set) + 1):
                    continue
                
                result = dfs(current_set + [x], new_candidates)
                if result:
                    return result
            
            return None
        
        # Initial candidates
        candidates = list(range(1, max_val + 1))
        nodes_visited[0] = 0
        result = dfs([], candidates)
        
        if result:
            return result
        
        # If failed, try a few randomized shuffles of the candidate list?
        # But the logic `candidates[idx+1:]` assumes sorted order for the "distinct and increasing" constraint 
        # which reduces symmetry. If we shuffle, we must sort the result or just return the set.
        # The problem doesn't require sorted output, just distinct integers.
        # However, the "increasing" constraint in DFS is just for symmetry breaking.
        # If we shuffle, we break the symmetry breaking, potentially exploring more states.
        # Better: Keep sorted, but maybe skip some small numbers?
        
        # Alternative: Random restart with different starting subsets?
        # Let's try one more time with a slightly different heuristic: 
        # Prioritize x that have many valid y's?
        # This is expensive.
        
        # Let's just rely on the next k (larger P) being easier.
        pass
        
    # Fallback: Return a valid set even if not optimal? 
    # The task says "minimize", but if we can't find 5 primes, we try 6, etc.
    # If we exhaust k=25, we should return something.
    # Unlikely to fail with k=25.
    
    # If we reach here, return a default set (will have high score, but valid)
    return list(range(1, n+1))

# EVOLVE-BLOCK-END
