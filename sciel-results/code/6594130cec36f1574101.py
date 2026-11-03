import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors and their indices
    # Vector is defined by (support_tuple, signs_tuple)
    # We map each unique vector to an integer ID.
    
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
    
    # 2. Build Conflict Graph
    # neighbors[i] = set/list of indices j that conflict with i
    # We need to check:
    # a) Same support: conflict if identical. (Actually, if we treat them as distinct nodes, identical vectors are the same node? No, we generated unique vectors. So same support, different signs are distinct nodes. Do they conflict?
    #    If $u \neq v$ and same support, $u \cdot v = 2a - 4$.
    #    If $a=3$ (1 sign diff), $u \cdot v = 2$. Dist$^2 = 4$. OK.
    #    If $a=2$ (2 sign diff), $u \cdot v = 0$. Dist$^2 = 8$. OK.
    #    So vectors on the same support NEVER conflict unless they are identical.
    #    Since we generated distinct sign patterns, there are NO conflicts within the same support.
    #    
    # b) Different support: Conflict if $|S \cap T| = 3$ AND signs agree on intersection.
    
    neighbors = [set() for _ in range(N)]
    
    # Precompute support intersections to speed up neighbor finding?
    # Or just iterate pairs of supports.
    # Supports: 330. Pairs: ~54k.
    # For each pair of supports (S1, S2) with intersection size 3:
    #   For each sign pattern in S1 (16) and S2 (16):
    #     Check if they agree on intersection.
    
    supports = list(support_map.keys())
    
    # Map support to list of vector indices
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute intersection info for support pairs
    # To save memory/time, we can just compute on the fly or precompute a list of conflicting (v1_idx, v2_idx) pairs.
    
    conflict_pairs = []
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue # Symmetry
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Find indices in s1 and s2 corresponding to inter
                # s1 is a tuple.
                # Map: index_in_tuple -> original_index
                # We need to check if signs match on inter.
                
                # Get the positions of the 3 common elements in s1 and s2
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        
                        # Check if signs match on the 3 positions
                        match = True
                        for p1, p2 in zip(pos_in_s1, pos_in_s2):
                            if signs1[p1] != signs2[p2]:
                                match = False
                                break
                        if match:
                            neighbors[idx1].add(idx2)
                            neighbors[idx2].add(idx1)
                            conflict_pairs.append((idx1, idx2))

    # 3. Backtracking Search for Max Independent Set
    # State:
    #   selected: set of selected indices
    #   candidates: set of indices that can still be selected (not selected, not excluded, not neighbor of selected)
    
    best_set = set()
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for final processing
    
    # Heuristic: Most Constrained Variable
    # We want to pick a candidate that has the most impact.
    # Usually, high degree in the *remaining* graph is good for branching (removes many candidates if selected).
    # However, calculating degrees dynamically is expensive.
    # Approximation: Use initial degree? Or just pick the first one?
    # Let's try picking the candidate with the highest number of neighbors in the current candidate set.
    
    def solve(candidates, selected):
        nonlocal best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if len(selected) > len(best_set):
                best_set = set(selected)
            return
        
        # Upper bound pruning
        if len(selected) + len(candidates) <= len(best_set):
            return
            
        # Heuristic: Pick vertex v in candidates with max degree within candidates
        # To speed this up, we might just pick the first one, or one with high global degree.
        # Let's try: Pick v with max len(neighbors[v] & candidates)
        # This is O(C * D) where C is candidates size.
        
        max_deg = -1
        v = -1
        # Sample if candidates is large?
        # For now, iterate all.
        for u in candidates:
            deg = 0
            # Count neighbors in candidates
            # neighbors[u] is a set
            # Intersection size
            # This is slow in Python.
            # Optimization: Precompute neighbor lists as lists?
            # Or just use global degree as a proxy?
            # Let's use global degree for now to save time, or a simple heuristic.
            # Actually, "Most Constrained" usually means fewest values left (in CSP).
            # In MIS, "Most Constrained" vertex is one that conflicts with many others,
            # so selecting it removes many options.
            # Let's approximate degree by len(neighbors[u]).
            
            # A better heuristic for MIS: Pick the vertex with the highest degree in the induced subgraph.
            # If we can't compute it fast, pick the one with highest global degree.
            
            if len(neighbors[u]) > max_deg:
                max_deg = len(neighbors[u])
                v = u
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates \ {v} \ neighbors[v]
        new_candidates_1 = set()
        for u in candidates:
            if u != v and u not in neighbors[v]:
                new_candidates_1.add(u)
        
        selected.add(v)
        solve(new_candidates_1, selected)
        selected.remove(v)
        
        # Branch 2: Exclude v
        # New candidates: candidates \ {v}
        new_candidates_2 = candidates - {v}
        solve(new_candidates_2, selected)

    # Initial candidates: all nodes
    initial_candidates = set(range(N))
    initial_selected = set()
    
    # To improve performance, we can order the initial candidates or use a different branching strategy.
    # The simple backtracking above might be too slow for 5280 nodes.
    # We can add a greedy initialization to get a good lower bound.
    
    # Greedy initialization
    greedy_selected = set()
    remaining = set(range(N))
    while remaining:
        # Pick node with min degree in remaining? Or max?
        # For greedy MIS, picking max degree is common.
        # Let's just pick the first one for speed in init.
        v = next(iter(remaining))
        greedy_selected.add(v)
        remaining -= neighbors[v]
        remaining.discard(v)
        
    best_set = set(greedy_selected)
    
    # Now run backtracking to improve
    # We need to be careful with recursion depth and time.
    # Python recursion limit might be an issue if depth is high, but width is the problem.
    # We will rely on the time limit.
    
    # To make the search more effective, we can use a "branch and bound" with a better bound.
    # Bound: len(selected) + len(candidates) is weak.
    # A tighter bound: len(selected) + sum of (1 / (1 + deg(u)/2))? No.
    # Simple bound is okay if we prune well.
    
    # Let's run the backtracking.
    # Note: The `solve` function above uses global degree for heuristic.
    # This might not be optimal but is fast.
    
    # One issue: `candidates` is a set. `candidates - {v}` creates a new set.
    # `new_candidates_1` construction is O(C).
    
    # To prevent stack overflow or excessive time, we can limit the depth or use iterative deepening?
    # Given the time limit, the recursive call will stop when time is up.
    
    try:
        solve(initial_candidates, initial_selected)
    except RecursionError:
        pass
        
    # Convert best_set to array
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = [all_vectors[i][3] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)
