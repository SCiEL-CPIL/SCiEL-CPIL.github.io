import time
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 255
    
    # Candidates: numbers in [1,255] with gcd(v,255)>1
    # i.e., divisible by 3, 5, or 17
    candidates = [v for v in range(1, n+1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    
    # Build adjacency
    m = len(candidates)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find max clique containing 255
    idx_255 = candidates.index(255)
    
    # Use greedy + local search approach
    # Start with all multiples of 3
    mult3 = set(i for i in range(m) if candidates[i] % 3 == 0)
    best_size = len(mult3)
    best_clique = mult3.copy()
    
    # Try to improve by adding elements
    # For each element not in the clique, check if it can be added
    # This is a local search
    
    # Actually, let me try a different strategy:
    # Try all "types" of cliques based on prime factor patterns
    
    # Type 1: All multiples of a single prime p dividing 255
    # p=3: 85, p=5: 51, p=17: 15
    
    # Type 2: Mixed - elements sharing different primes
    # e.g., elements divisible by 3 or 5, but with constraints
    
    # Let me try: all elements divisible by 3, plus elements divisible by 5 
    # that are also divisible by some other prime in common with all multiples of 3
    # This is hard.
    
    # Let me just do a maximum clique search with time limit
    # Using a simple recursive approach with pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    best = mult3.copy()
    best_sz = len(best)
    
    def try_add(clique, candidates_set):
        nonlocal best, best_sz
        if time.time() > deadline - 5:
            return
        if not candidates_set:
            if len(clique) > best_sz:
                best_sz = len(clique)
                best = clique.copy()
            return
        
        # Pick a candidate
        v = next(iter(candidates_set))
        candidates_set.remove(v)
        
        # Option 1: Don't include v
        try_add(clique, candidates_set)
        
        # Option 2: Include v (if compatible with all in clique)
        if all(v in adj[u] for u in clique):
            clique.add(v)
            # New candidates must be compatible with v
            new_cands = candidates_set & adj[v]
            try_add(clique, new_cands)
            clique.remove(v)
    
    # This is too slow for 127 vertices. Let me use a better algorithm.
    
    # Use maximum clique with pivot (Bron-Kerbosch)
    # But limit to reasonable time
    
    bk_best = mult3.copy()
    bk_best_sz = len(bk_best)
    
    def bk(R, P, X):
        nonlocal bk_best, bk_best_sz
        if time.time() > deadline - 5:
            return
        if not P and not X:
            if len(R) > bk_best_sz:
                bk_best_sz = len(R)
                bk_best = R.copy()
            return
        # Pivot
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            Q = P - adj[u]
        else:
            Q = set()
        for v in list(Q):
            if v not in P:
                continue
            bk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    # Must include 255
    P0 = adj[idx_255].copy()
    bk({idx_255}, P0, set())
    
    if bk_best_sz > best_sz:
        best = bk_best
        best_sz = bk_best_sz
    
    result = sorted([candidates[i] for i in best])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1
    
    return result
