"""Improved program: deterministic minimal antichain search with precomputed divisibility."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 75
_START = time.time()

def _precompute_smooths(limit):
    smooths = []
    for x in range(1, limit + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooths.append(x)
    return smooths

def _solve_target(n, smooths, divs, index_map):
    # Filter smooths <= n
    pool = [x for x in smooths if x <= n]
    if not pool:
        return None
    
    pool_set = set(pool)
    
    # k=1
    if n in pool_set:
        return [n]
        
    # k=2
    # Find a, b in pool, a < b, a+b=n, a not divide b
    # Since a+b=n, b=n-a.
    # We need a < n-a => 2a < n => a < n/2.
    # And b in pool.
    # And not (b % a == 0).
    for a in pool:
        if a * 2 >= n: # a must be smaller than b, so a < n/2
            break
        b = n - a
        if b in pool_set:
            if b % a != 0:
                return [a, b]
                
    # k >= 3
    # Use combinations. Pool size is small.
    # Sort pool to help with early termination? Not strictly needed for combinations.
    m = len(pool)
    max_k = min(30, m)
    
    for k in range(3, max_k + 1):
        # Optional: Pruning based on min/max sum?
        # Min sum of k distinct smallest: sum(pool[:k])
        # Max sum of k distinct largest: sum(pool[-k:])
        # If n not in [min_sum, max_sum], skip? 
        # But combinations iterates all. We can just iterate.
        
        # To speed up, we can check antichain constraint efficiently.
        # Precomputed divs matrix: divs[i][j] is True if smooths[i] divides smooths[j].
        # We need indices of pool elements.
        pool_indices = [index_map[x] for x in pool]
        
        for combo in combinations(range(m), k):
            # Check sum
            s = 0
            for idx in combo:
                s += pool[idx]
            if s != n:
                continue
            
            # Check antichain
            valid = True
            for i in range(k):
                for j in range(i + 1, k):
                    gi = pool_indices[combo[i]]
                    gj = pool_indices[combo[j]]
                    # Check if pool[combo[i]] divides pool[combo[j]] or vice versa
                    # divs[gi][gj] or divs[gj][gi]
                    if divs[gi][gj] or divs[gj][gi]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return [pool[i] for i in combo]
                
    return None

def construct():
    global _START
    _START = time.time()
    
    # Precompute
    smooths = _precompute_smooths(128)
    m = len(smooths)
    index_map = {x: i for i, x in enumerate(smooths)}
    
    # Precompute divisibility matrix
    # divs[i][j] = True if smooths[i] divides smooths[j]
    divs = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if smooths[j] % smooths[i] == 0:
                divs[i][j] = True
                
    answers = [None] * 128
    
    for n in range(1, 129):
        if time.time() - _START > _DEADLINE:
            break
        ans = _solve_target(n, smooths, divs, index_map)
        if ans is None:
            # Fallback: should not happen for 1..128, but just in case
            # Try to find any representation, even non-minimal?
            # Or use [n] if smooth?
            if n in index_map:
                ans = [n]
            else:
                # Last resort: brute force with larger k or random?
                # Given the problem constraints, a solution should exist.
                # If _solve_target failed, it means no antichain found up to max_k.
                # This is unlikely for n<=128.
                ans = [] 
        answers[n - 1] = ans
        
    return answers
# EVOLVE-BLOCK-END
