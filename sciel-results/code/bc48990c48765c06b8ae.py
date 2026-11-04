import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Strategy: Find the maximum independent set of 4-subsets of {0..10}
    # such that any two subsets intersect in at most 2 elements.
    # Then return all 16 sign variations for each subset.
    
    supports4 = list(combinations(range(D), 4))
    N4 = len(supports4)
    sup4_sets = [set(s) for s in supports4]
    
    # Build conflict graph
    # Conflict if intersection >= 3
    adj4 = [set() for _ in range(N4)]
    for i in range(N4):
        for j in range(i + 1, N4):
            if len(sup4_sets[i] & sup4_sets[j]) >= 3:
                adj4[i].add(j)
                adj4[j].add(i)
                
    def greedy_mis(adj, n, rng, deadline):
        selected = set()
        avail = set(range(n))
        while avail:
            if time.time() > deadline:
                break
            # Find node with min degree in avail
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(adj[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            u = rng.choice(candidates)
            selected.add(u)
            avail -= {u} | (adj[u] & avail)
        return selected

    def local_search_mis(adj, n, iset, rng, deadline):
        iset = set(iset)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try 1-1 swap: remove v, add w
            # Try 1-2 swap: remove v, add w1, w2
            for v in list(iset):
                if time.time() > deadline:
                    break
                sel_minus_v = iset - {v}
                potential = []
                for x in range(n):
                    if x in iset or x == v:
                        continue
                    if not (adj[x] & sel_minus_v):
                        potential.append(x)
                
                # 1-1 swap
                for w in potential:
                    if w not in adj[v]: # This check is not quite right for 1-1
                        pass # We just need w to be compatible with sel_minus_v, which it is.
                        # But we are replacing v with w.
                        # Size doesn't change. No gain.
                        # So skip 1-1.
                        
                # 1-2 swap
                L = len(potential)
                found = False
                for i in range(L):
                    u1 = potential[i]
                    for j in range(i + 1, L):
                        u2 = potential[j]
                        if u2 not in adj[u1]:
                            iset.remove(v)
                            iset.add(u1)
                            iset.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
        return iset

    best_iset4 = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    mis_deadline = start_time + 100.0
    for r in range(200):
        if time.time() > mis_deadline:
            break
        iset = greedy_mis(adj4, N4, rng, mis_deadline)
        iset = local_search_mis(adj4, N4, iset, rng, mis_deadline)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset4 = iset.copy()
            
    # Generate vectors
    vectors = []
    for i in best_iset4:
        sup = supports4[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    # Fallback: If we didn't find many, try weight-2
    # But weight-4 should be better if best_size > 13 (13*16=208 < 220)
    # If best_size < 14, we might be better off with weight-2.
    # Let's calculate the weight-2 count: 220.
    # If 16 * best_size < 220, i.e., best_size < 13.75, i.e., best_size <= 13.
    # Then use weight-2.
    
    if 16 * best_size < 220:
         # Generate all weight-2 vectors
         supports2 = list(combinations(range(D), 2))
         for S2 in supports2:
             for signs in product([1, -1], repeat=2):
                 v = np.zeros(D, dtype=np.int8)
                 for idx, s in zip(S2, signs):
                     v[idx] = s
                 vectors.append(v)
                 # Note: We don't need to check compatibility if we use ONLY weight-2.
                 # We verified earlier that all 220 are mutually compatible.
    else:
         # Use the weight-4 vectors found
         pass

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
