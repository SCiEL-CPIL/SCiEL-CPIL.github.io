import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    D = 11
    
    # Generate weight-4 vectors
    # Support: combinations of 4 indices
    # Signs: 16 combinations of +/- 1
    candidates_list = []
    supports_list = []
    
    supports = list(itertools.combinations(range(D), 4))
    # Map support to index for fast lookup
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            candidates_list.append(v)
            
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates) # 5280
    
    # Build conflict graph
    # Conflict if dot > 2
    # We can compute dot products on the fly or precompute.
    # Precomputing 5280x5280 int8 matrix is 27MB, which is fine.
    # But we only need the neighbors.
    
    # To speed up, let's use the support structure.
    # For each vector i, find neighbors j.
    # i has support S_i.
    # Neighbors j must have support S_j such that |S_i \cap S_j| == 3.
    # And signs must match on the intersection.
    
    # Precompute neighbors for each vector
    neighbors = [[] for _ in range(n)]
    
    # Index vectors by support
    # vectors_by_support[support_idx] = list of vector indices
    vectors_by_support = {i: [] for i in range(len(supports))}
    for idx, v in enumerate(candidates_list):
        # Find support
        non_zero = tuple(i for i in range(D) if v[i] != 0)
        s_idx = support_to_idx[non_zero]
        vectors_by_support[s_idx].append(idx)
        
    # For each support, precompute the "sign pattern" on any 3-subset?
    # Actually, let's just iterate.
    
    # For each pair of supports (S1, S2) with intersection size 3:
    # Iterate through vectors in S1 and S2.
    # Check if they conflict.
    
    # Precompute support pairs with intersection 3
    support_pairs = []
    num_supports = len(supports)
    for i in range(num_supports):
        s1 = set(supports[i])
        for j in range(i + 1, num_supports):
            s2 = set(supports[j])
            if len(s1 & s2) == 3:
                support_pairs.append((i, j))
                
    # For each pair of supports, add conflicts
    for s1_idx, s2_idx in support_pairs:
        s1 = set(supports[s1_idx])
        s2 = set(supports[s2_idx])
        intersection = s1 & s2
        inter_list = list(intersection)
        
        vecs1 = vectors_by_support[s1_idx]
        vecs2 = vectors_by_support[s2_idx]
        
        # For each vector in vecs1, determine its sign pattern on intersection
        # We need to check if vec2 has the SAME signs on intersection.
        
        # Optimization: Group vecs2 by their sign pattern on the intersection
        # There are 2^3 = 8 possible sign patterns.
        # pattern: tuple of signs at inter_list[0], inter_list[1], inter_list[2]
        
        vecs2_by_pattern = {}
        for v2_idx in vecs2:
            v2 = candidates_list[v2_idx]
            pat = tuple(v2[pos] for pos in inter_list)
            if pat not in vecs2_by_pattern:
                vecs2_by_pattern[pat] = []
            vecs2_by_pattern[pat].append(v2_idx)
            
        for v1_idx in vecs1:
            v1 = candidates_list[v1_idx]
            pat1 = tuple(v1[pos] for pos in inter_list)
            # Conflict if v2 has the same pattern
            if pat1 in vecs2_by_pattern:
                for v2_idx in vecs2_by_pattern[pat1]:
                    neighbors[v1_idx].append(v2_idx)
                    neighbors[v2_idx].append(v1_idx)
                    
    # Now we have the conflict graph.
    # Run Greedy + Swap Local Search
    
    def greedy_max_independent_set(order):
        selected = []
        selected_set = set()
        available = [True] * n
        
        for idx in order:
            if not available[idx]:
                continue
            # Check if idx conflicts with any selected
            # Since we remove neighbors from 'available', if available[idx] is True,
            # it means it doesn't conflict with any *previously selected* vertex?
            # No, standard greedy:
            # Pick idx. Add to selected.
            # For all neighbors of idx, set available[neighbor] = False.
            selected.append(idx)
            selected_set.add(idx)
            for nb in neighbors[idx]:
                available[nb] = False
                
        return selected

    def local_search_swap(selected_list, time_limit):
        selected = list(selected_list)
        selected_set = set(selected)
        
        # Precompute: for each unselected node, is it compatible with all selected?
        # This is expensive to recompute.
        # Instead, maintain a count of conflicts with selected for each node.
        # conflict_count[j] = number of neighbors of j that are in selected.
        # j is addable if conflict_count[j] == 0 and j not in selected.
        
        conflict_count = np.zeros(n, dtype=np.int32)
        for s in selected:
            for nb in neighbors[s]:
                conflict_count[nb] += 1
                
        # Initial addable
        addable = [i for i in range(n) if i not in selected_set and conflict_count[i] == 0]
        
        # Greedy add
        while addable and time.time() < time_limit:
            # Pick one with min degree in the remaining graph?
            # Or just pick the first one?
            # To be safe, pick one that has fewest neighbors in the *unselected* pool?
            # Just pick random or first.
            idx = addable.pop()
            selected.append(idx)
            selected_set.add(idx)
            # Update conflict counts
            for nb in neighbors[idx]:
                if nb not in selected_set:
                    conflict_count[nb] += 1
                    if conflict_count[nb] == 1:
                        # It just became non-addable
                        try:
                            addable.remove(nb)
                        except ValueError:
                            pass
            # Remove idx from addable if it was there (it shouldn't be, but just in case)
            # Also, removing idx might make its neighbors... no, adding idx only increases conflict counts.
            # So we only remove from addable.
            
        # Swap Local Search
        # Try to swap one selected node for two unselected nodes.
        # This is complex.
        # Alternative: Simple 1-swap (replace 1 with 1) doesn't help size.
        # 2-for-1 swap: Remove s, add u, v.
        # u, v must be compatible with (selected - {s}) and with each other.
        # And u, v must conflict with s (otherwise we would have just added them).
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle selected to try different orders
            random.shuffle(selected)
            
            for s in selected:
                if time.time() > time_limit:
                    break
                    
                # Find candidates u, v that:
                # 1. Are not in selected.
                # 2. Are compatible with all selected except s.
                #    i.e., conflict_count[u] <= 1 (and the 1 conflict is s)
                #    Actually, if u conflicts with s, and no other selected, then conflict_count[u] == 1.
                #    If u doesn't conflict with s, conflict_count[u] == 0. But then u would have been added in the greedy phase.
                #    So we look for u with conflict_count[u] == 1 and s in neighbors[u].
                
                # Let's find all such u
                candidates_u = []
                for u in range(n):
                    if u in selected_set:
                        continue
                    if conflict_count[u] == 1 and s in neighbors[u]:
                        candidates_u.append(u)
                        
                if len(candidates_u) < 2:
                    continue
                    
                # Find a pair u, v in candidates_u that are compatible (no edge between them)
                # And compatible with each other.
                # Since the graph is sparse, we can check pairs.
                # To speed up, we can just pick a random subset of candidates_u.
                
                # Sort by degree?
                # Just try a few pairs
                random.shuffle(candidates_u)
                found = False
                for i in range(min(10, len(candidates_u))):
                    u = candidates_u[i]
                    # Check if any other v in candidates_u is compatible with u
                    for j in range(i + 1, min(10, len(candidates_u))):
                        v = candidates_u[j]
                        # Check if u and v are connected
                        if v not in neighbors[u]:
                            # Do the swap
                            # Remove s
                            selected.remove(s)
                            selected_set.discard(s)
                            
                            # Add u, v
                            selected.append(u)
                            selected_set.add(u)
                            selected.append(v)
                            selected_set.add(v)
                            
                            # Update conflict counts
                            # This is tricky to do incrementally correctly without a full recompute
                            # Given n=5280, a full recompute of conflict_count is O(E) ~ 150k ops.
                            # 150k ops is fast (<1ms).
                            # Let's just recompute conflict_count from scratch.
                            conflict_count = np.zeros(n, dtype=np.int32)
                            for sel in selected:
                                for nb in neighbors[sel]:
                                    conflict_count[nb] += 1
                                    
                            # Rebuild addable list
                            addable = [i for i in range(n) if i not in selected_set and conflict_count[i] == 0]
                            
                            # Greedy add again
                            while addable and time.time() < time_limit:
                                idx = addable.pop()
                                selected.append(idx)
                                selected_set.add(idx)
                                for nb in neighbors[idx]:
                                    if nb not in selected_set:
                                        conflict_count[nb] += 1
                                        if conflict_count[nb] == 1:
                                            try:
                                                addable.remove(nb)
                                            except ValueError:
                                                pass
                            
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
                    
        return selected

    best_size = 0
    best_result = []
    
    np.random.seed(42)
    
    # Multiple restarts
    for trial in range(100):
        if time.time() > deadline - 10.0:
            break
            
        # Random ordering
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy
        sel = greedy_max_independent_set(order)
        
        # Local Search
        sel = local_search_swap(sel, deadline - 5.0)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = sel
            
    if not best_result:
        # Fallback: just return first vector
        return candidates[0].reshape(1, -1)
        
    return candidates[np.array(best_result, dtype=np.int64)].astype(np.int64)
