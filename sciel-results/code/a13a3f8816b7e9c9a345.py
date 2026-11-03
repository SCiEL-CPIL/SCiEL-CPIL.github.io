import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors and their indices
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph using Bitmasks
    # neighbors[i] is a big integer bitmask where bit j is 1 if i conflicts with j
    
    neighbors = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute support intersections
    # We only care about pairs with intersection size 3
    # Map support to its index in the supports list for fast lookup?
    # Or just iterate.
    
    for i_s, s1 in enumerate(supports):
        vecs_s1 = sup_to_vecs[s1]
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Find positions in s1 and s2 for the intersection elements
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s2 = sup_to_vecs[s2]
                
                # For each pair of vectors, check if they conflict
                # Optimization: Group vectors by their sign pattern on the intersection
                # There are 8 possible sign patterns on the 3 intersection elements.
                # Map pattern (tuple of 3 signs) to list of vector indices
                
                map_s1 = {}
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    pat = tuple(signs1[p] for p in pos_in_s1)
                    if pat not in map_s1:
                        map_s1[pat] = []
                    map_s1[pat].append(idx1)
                    
                map_s2 = {}
                for idx2 in vecs_s2:
                    signs2 = all_vectors[idx2][2]
                    pat = tuple(signs2[p] for p in pos_in_s2)
                    if pat not in map_s2:
                        map_s2[pat] = []
                    map_s2[pat].append(idx2)
                    
                # Conflicts occur if patterns are identical
                for pat, list1 in map_s1.items():
                    if pat in map_s2:
                        list2 = map_s2[pat]
                        for idx1 in list1:
                            mask1 = 0
                            for idx2 in list2:
                                mask1 |= (1 << idx2)
                            neighbors[idx1] |= mask1
                            # Also add reverse edges
                            for idx2 in list2:
                                neighbors[idx2] |= (1 << idx1)
                                # Wait, this is inefficient. 
                                # Better: 
                                # for idx1 in list1:
                                #    for idx2 in list2:
                                #       neighbors[idx1] |= (1 << idx2)
                                #       neighbors[idx2] |= (1 << idx1)
                                
    # 3. Greedy Initial Solution
    best_set = set()
    candidates = (1 << N) - 1
    while candidates:
        # Find vertex with max degree in candidates
        max_deg = -1
        v = -1
        c = candidates
        while c:
            lsb = c & -c
            u = lsb.bit_length() - 1
            deg = (candidates & neighbors[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            c ^= lsb
        
        if v == -1: break
        best_set.add(v)
        candidates &= ~(neighbors[v] | (1 << v))
        
    best_size = len(best_set)
    
    # 4. Branch and Bound Search
    start_time = time.time()
    time_limit = 110.0
    
    # We need to store the actual set of vertices for the best solution found
    best_solution = list(best_set)
    
    def solve(candidates, selected, selected_list):
        nonlocal best_size, best_solution, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Upper bound: current size + number of candidates
        # This is weak. Let's use a slightly tighter bound if possible?
        # For now, stick to this but rely on fast bitmask ops.
        if len(selected) + candidates.bit_count() <= best_size:
            return
            
        if not candidates:
            if len(selected) > best_size:
                best_size = len(selected)
                best_solution = list(selected_list)
            return
            
        # Heuristic: Pick vertex with max degree in candidates
        max_deg = -1
        v = -1
        c = candidates
        while c:
            lsb = c & -c
            u = lsb.bit_length() - 1
            deg = (candidates & neighbors[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            c ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        new_candidates_1 = candidates & ~(neighbors[v] | (1 << v))
        selected.add(v)
        selected_list.append(v)
        solve(new_candidates_1, selected, selected_list)
        selected_list.pop()
        selected.remove(v)
        
        # Branch 2: Exclude v
        new_candidates_2 = candidates & ~(1 << v)
        solve(new_candidates_2, selected, selected_list)

    initial_candidates = (1 << N) - 1
    initial_selected = set()
    initial_list = []
    
    try:
        solve(initial_candidates, initial_selected, initial_list)
    except RecursionError:
        pass
        
    # Convert best_solution to array
    if not best_solution:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = [all_vectors[i][3] for i in best_solution]
    return np.array(result_vectors, dtype=np.int64)
