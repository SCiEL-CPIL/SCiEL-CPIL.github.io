import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute LCMs and Bad Triples
    lcm_cache = {}
    def get_lcm(a, b):
        if a > b: a, b = b, a
        key = (a, b)
        if key not in lcm_cache:
            lcm_cache[key] = lcm(a, b)
        return lcm_cache[key]

    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l_ab = get_lcm(a, b)
            for c in range(b + 1, N + 1):
                l_ac = get_lcm(a, c)
                if l_ac != l_ab:
                    continue
                l_bc = get_lcm(b, c)
                if l_bc == l_ab:
                    bad_triples.append((a, b, c))
    
    # 2. Precompute conflict structures
    conflict_pairs = {x: [] for x in elements}
    for a, b, c in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
        
    def can_add(x, S):
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True

    # 3. Search Strategy
    best_set = set()
    best_size = 0
    
    # Greedy Randomized Construction
    while time.time() < deadline - 10.0:
        S = set()
        candidates = elements.copy()
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # 4. Local Search / Improvement on best_set
    # Pass 1: Greedy Add
    S = best_set.copy()
    improved = True
    while improved:
        improved = False
        for x in elements:
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
        
    # Pass 2: 1-for-2 Swaps
    if time.time() < deadline - 2.0:
        S = best_set.copy()
        for u in list(best_set):
            if u not in S:
                continue
            if time.time() > deadline - 1.0:
                break
                
            S.remove(u)
            added = []
            
            for v in elements:
                if v == u or v in S:
                    continue
                if can_add(v, S):
                    S.add(v)
                    added.append(v)
                    
            if len(added) >= 2:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
            else:
                for v in added:
                    S.discard(v)
                S.add(u)
                
    # Final Pass 1 again
    S = best_set.copy()
    improved = True
    while improved:
        improved = False
        for x in elements:
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()

    if not best_set:
        best_set = {1}
        
    return list(best_set)
