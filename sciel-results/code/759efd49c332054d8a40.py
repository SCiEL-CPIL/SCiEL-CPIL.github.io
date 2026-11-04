import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Strategy: Use weight-4 supports.
    # For a support of size w=4, there are 2^4=16 sign patterns.
    # The minimum distance squared between two distinct sign patterns in the same support is:
    # dist^2 = ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v
    # u.v is the dot product. For binary signs, u.v = sum(s_i t_i).
    # The minimum distance occurs when they differ in the most positions? No.
    # u.v is maximized when signs are most similar.
    # If u and v differ in k positions, u.v = 4 - 2k.
    # dist^2 = 8 - 2(4 - 2k) = 8 - 8 + 4k = 4k.
    # Minimum k=1 (differ in 1 position) -> dist^2 = 4.
    # So if we take ALL sign patterns for a support, the min intra-support distance is 4.
    # This means we can set max_norm^2 = 4.
    # Then we need all pairwise distances >= 4.
    
    # For two different supports S, T:
    # dist^2 = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4 => 4 >= 2 u.v => u.v <= 2.
    # u.v = sum_{i in S∩T} s_i t_i.
    # Let I = S∩T, |I| = m.
    # u.v can range from -m to m in steps of 2.
    # We need u.v <= 2.
    
    # If m=0: u.v=0. OK.
    # If m=1: u.v=±1. OK.
    # If m=2: u.v=-2,0,2. OK (max 2).
    # If m=3: u.v=-3,-1,1,3. We need u.v <= 2. So u.v=3 is forbidden.
    #    u.v=3 means all signs in intersection are identical.
    # If m=4: same support. Handled above.
    
    # So, two supports S, T with |S∩T|=3 are compatible ONLY IF we do not choose sign patterns
    # that agree on all 3 intersection indices.
    # This complicates things. We can't just take all sign patterns for arbitrary supports.
    
    # However, if we restrict to a set of supports such that |S∩T| <= 2 for all pairs,
    # then we can include ALL 16 sign patterns for each support.
    # This reduces to finding a Maximum Independent Set (MIS) in the graph where
    # nodes are 4-subsets of [11], and edges connect subsets with intersection size >= 3.
    
    # This is exactly the strategy from Inspiration 1, which scored 2.5455.
    # The current program scores 1.0 (220 vectors).
    # Inspiration 1 found ~560 vectors (2.5455 * 220 approx? No, seed is 220. 2.5455 * 220 = 560).
    # Let's verify the score logic.
    # Score = k / seed_k.
    # If seed is 220, and score is 2.5455, then k = 220 * 2.5455 ≈ 560.
    # 16 * 35 = 560. So it found 35 supports.
    
    # Let's implement the MIS search on 4-subsets with intersection >= 3.
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph
    # incompat[i, j] = True if |sup_i ∩ sup_j| >= 3
    neighbors = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 10.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    def greedy_min_degree(available, rng):
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 5.0:
                break
            
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            u = rng.choice(candidates)
            selected.add(u)
            
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected

    def local_search(selected, rng):
        """
        Try to improve the independent set using 1-2 swaps.
        """
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try 1-2 swap: remove v, add u1, u2
            for v in list(selected):
                if time.time() > deadline - 5.0:
                    break
                
                # Find all nodes compatible with selected \ {v}
                # A node x is compatible if x is not in neighbors[y] for any y in selected \ {v}
                # And x not in selected
                
                # Optimization: precompute blocked set?
                # blocked = union(neighbors[y] for y in selected \ {v})
                # But this is O(|selected| * avg_deg) per v.
                # |selected| ~ 35. avg_deg ~ 10. 350 ops.
                # N_sup = 330.
                # Checking all x in N_sup is 330 * 350 = 115k ops per v.
                # 35 v's -> 4M ops per iteration.
                # This might be too slow for many iterations.
                
                # Better:
                # potential_adds = [x for x in range(N_sup) if x not in selected and x != v and not (neighbors[x] & (selected - {v}))]
                
                # Let's just do the check directly but efficiently.
                # We can precompute the "blocked" set for the current selected set once?
                # No, it changes when we remove v.
                
                # Alternative:
                # For each x not in selected, check if it conflicts with any y in selected.
                # If it conflicts with v, it might be a candidate if we remove v.
                # If it conflicts with any other y, it's blocked.
                
                # Let's iterate x in range(N_sup):
                #   if x in selected or x == v: continue
                #   conflicts_with_others = False
                #   for y in selected:
                #       if y == v: continue
                #       if x in neighbors[y]:
                #           conflicts_with_others = True
                #           break
                #   if not conflicts_with_others:
                #       potential_adds.append(x)
                
                # This is O(N * |S|) per v.
                # 330 * 35 = 11k ops per v.
                # 35 v's -> 400k ops per full pass.
                # This is fast enough.
                
                potential_adds = []
                sel_minus_v = selected - {v}
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    # Check if x is compatible with all y in sel_minus_v
                    # i.e., x not in neighbors[y] for all y in sel_minus_v
                    # Equivalent: neighbors[x] & sel_minus_v is empty
                    if not (neighbors[x] & sel_minus_v):
                        potential_adds.append(x)
                        
                # Find pair u1, u2 in potential_adds that are compatible
                # i.e., u1 not in neighbors[u2]
                found = False
                L = len(potential_adds)
                for i in range(L):
                    u1 = potential_adds[i]
                    for j in range(i + 1, L):
                        u2 = potential_adds[j]
                        if u2 not in neighbors[u1]:
                            # Swap
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if improved:
                continue
        return selected

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        avail = set(range(N_sup))
        iset = greedy_min_degree(avail, rng)
        iset = local_search(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
        # If we hit a high number, we can stop early?
        # Theoretical max:
        # Each support has 4 elements.
        # Total pairs covered: 330 supports * 6 pairs = 1980 pair occurrences.
        # Total pairs in [11]: C(11,2) = 55.
        # If we restrict to intersection <= 2, no pair is repeated?
        # No, intersection <= 2 means a pair can appear in multiple supports?
        # If S and T share a pair {a,b}, then |S∩T| >= 2.
        # If |S∩T|=2, they share exactly one pair.
        # So a pair can be in multiple supports as long as they don't share 3 elements.
        # This is not a packing of pairs.
        
    if best_iset is None:
        return np.array([[]], dtype=np.int8)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
