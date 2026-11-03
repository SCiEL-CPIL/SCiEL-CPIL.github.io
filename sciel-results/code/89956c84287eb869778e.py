import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Generate all weight-4 vectors
    # Supports: combinations of 4 indices from 0..10
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Map support index to support tuple
    # Map (support_idx, signs_tuple) to vector_index
    # Total vectors = n_sup * 16
    
    vectors = []
    sup_indices = []
    sign_tuples = []
    
    for s_idx, sup in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            sup_indices.append(s_idx)
            sign_tuples.append(signs)
            
    N = len(vectors)
    
    # Build conflict graph
    # Conflict if dist^2 < 4
    # As analyzed:
    # 1. Same support: distinct vectors always valid (dist^2 >= 4).
    #    So no conflicts between distinct vectors of same support.
    # 2. Intersection 3: Conflict if signs match on the 3 shared indices.
    # 3. Intersection < 3: No conflict.
    
    # We need to efficiently find neighbors.
    # For each vector i, find j != i such that:
    # - |S_i \cap S_j| == 3
    # - signs match on S_i \cap S_j
    
    # Optimization:
    # Group vectors by support.
    # For each support s, and each subset I of size 3 of s:
    #   The vectors in support s with signs restricted to I are "keys".
    #   Any vector j with support s' containing I will conflict with i if 
    #   signs of j on I match signs of i on I.
    
    # Let's build a dictionary:
    # key: (tuple(sorted(I)), signs_on_I)
    # value: list of vector indices having this support-subset-signs combination.
    
    # Actually, for a vector i with support S and signs sig:
    # For each I subset of S (size 3):
    #   sig_I = signs of i on I.
    #   Any other vector j with support S' containing I and signs sig'_I == sig_I conflicts.
    
    # We can iterate over all pairs? N=5280. N^2 = 28M.
    # 28M distance checks in Python might be slow (120s limit).
    # But we can use the grouping method.
    
    # Method:
    # 1. Create a dict `lookup` mapping (I_tuple, signs_tuple_on_I) to list of vector indices.
    # 2. For each vector i, for each I in subsets(S_i, 3):
    #    key = (I, signs_i_on_I)
    #    candidates = lookup[key]
    #    For each j in candidates:
    #       if j != i and sup_indices[j] != sup_indices[i]: # Same support is not conflict
    #          add edge (i, j)
    
    # Note: If j has same support as i, they are not in conflict (as distinct vectors).
    # But they might appear in the same lookup bucket?
    # Yes, if they have same signs on I.
    # So we must filter out j with same support.
    
    lookup = {}
    
    # Precompute subsets of each support
    sup_subsets = {}
    for s_idx, sup in enumerate(supports):
        subs = list(itertools.combinations(sup, 3))
        sup_subsets[s_idx] = subs
        
    # Populate lookup
    for i in range(N):
        s_idx = sup_indices[i]
        signs = sign_tuples[i]
        # Map sign tuple to specific indices in the support
        # signs[k] corresponds to supports[s_idx][k]
        for I in sup_subsets[s_idx]:
            # Get signs corresponding to indices in I
            # I is a tuple of coordinate indices, e.g., (0, 1, 2)
            # We need the signs at these coordinates.
            # Since supports[s_idx] is sorted, and I is a subset, 
            # we can find the positions of I in supports[s_idx].
            
            # To make this fast, precompute the mapping from coordinate index to position in support.
            # Or just iterate.
            # Since size is small, let's just extract.
            sig_I = []
            for coord in I:
                # Find position of coord in supports[s_idx]
                # This is O(4)
                pos = supports[s_idx].index(coord)
                sig_I.append(signs[pos])
            sig_I = tuple(sig_I)
            
            key = (I, sig_I)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(i)
            
    # Build adjacency list
    adj = [set() for _ in range(N)]
    
    for i in range(N):
        s_idx = sup_indices[i]
        signs = sign_tuples[i]
        for I in sup_subsets[s_idx]:
            sig_I = []
            for coord in I:
                pos = supports[s_idx].index(coord)
                sig_I.append(signs[pos])
            sig_I = tuple(sig_I)
            
            key = (I, sig_I)
            candidates = lookup.get(key, [])
            for j in candidates:
                if j == i:
                    continue
                # Check if same support
                if sup_indices[j] == s_idx:
                    continue
                # If we reach here, j has different support, shares I, and matches signs on I.
                # This is a conflict.
                adj[i].add(j)
                adj[j].add(i)
                
    # Now we have the conflict graph.
    # Find Maximum Independent Set.
    
    # Greedy Initialization
    def greedy_is():
        remaining = set(range(N))
        current_set = set()
        # Sort by degree to pick low degree first?
        # Or just pick min degree in remaining.
        # To speed up, we can precompute degrees.
        degrees = [len(adj[i]) for i in range(N)]
        
        # We will use a priority queue or just scan.
        # Scanning N=5280 is fine.
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                # Degree in remaining
                # We can approximate by original degree, or compute exactly.
                # Exact is safer for quality.
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # Local Search (Simulated Annealing)
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.999
    steps = 50000
    
    # Precompute neighbors list for faster access? adj is list of sets.
    
    for step in range(steps):
        # Pick random v
        v = random.randint(0, N - 1)
        
        if v in S:
            continue
            
        # v is not in S
        N_neigh = adj[v].intersection(S)
        
        if len(N_neigh) == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - len(N_neigh)
            
            if delta > 0:
                S.difference_update(N_neigh)
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                if delta == 0:
                    if random.random() < 0.5:
                        S.difference_update(N_neigh)
                        S.add(v)
                else:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S.difference_update(N_neigh)
                        S.add(v)
                        current_size += delta
                        
        T *= cooling
        
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
            
    # Construct vectors from best_set
    result_vectors = [vectors[i] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)
