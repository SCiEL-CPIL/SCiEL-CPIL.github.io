"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def find_conflicts(elements):
        """Find all triples (a, b, c) such that a + b = c."""
        val_to_idx = {v: i for i, v in enumerate(elements)}
        conflicts = []
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    conflicts.append((i, j, k))
        return conflicts

    def greedy_hitting_set(elements, conflicts, seed):
        """
        Greedy algorithm to remove minimum elements to break all conflicts.
        Returns the set of remaining elements.
        """
        if not conflicts:
            return elements[:]
        
        # Map element index to set of conflict indices it participates in
        elem_to_conflicts = defaultdict(set)
        for idx, (i, j, k) in enumerate(conflicts):
            elem_to_conflicts[i].add(idx)
            elem_to_conflicts[j].add(idx)
            elem_to_conflicts[k].add(idx)
            
        # Active elements
        active = set(range(len(elements)))
        # Active conflicts
        active_conflicts = set(range(len(conflicts)))
        
        rng = random.Random(seed)
        
        while active_conflicts:
            # Pick a conflict to resolve? No, pick an element to remove.
            # We want to remove the element that covers the most active conflicts.
            # Calculate coverage for each active element.
            # Optimization: Instead of recalculating for all, we can pick a random active conflict,
            # and choose the best element from that conflict to remove.
            
            # To be more effective, let's iterate.
            # For efficiency, we can just pick the element with max degree in the active conflict hypergraph.
            # Recalculating degrees for all active elements is O(E_active).
            # With N=240, E is roughly N^2. This might be slow if done naively in Python loops.
            
            # Let's try a simpler heuristic:
            # Pick a random active conflict (i, j, k).
            # Among i, j, k, pick the one that is in the most OTHER active conflicts.
            # Remove it.
            
            conflict_idx = rng.choice(list(active_conflicts))
            i, j, k = conflicts[conflict_idx]
            
            candidates = [i, j, k]
            # Filter to active elements
            candidates = [c for c in candidates if c in active]
            
            if not candidates:
                # Should not happen if we maintain consistency
                active_conflicts.remove(conflict_idx)
                continue
                
            # Score candidates by number of active conflicts they are in
            # We need to count how many active conflicts each candidate is in.
            # We can maintain a counter array `conf_count` for each element.
            
            # Let's restructure to maintain counts.
            pass # This logic inside the loop is messy. Let's use the global degree approach.

        # The previous logic was getting complex. Let's use a simpler, robust approach:
        # Iterative removal based on global degree.
        
        # Reset
        active = set(range(len(elements)))
        active_conflicts = set(range(len(conflicts)))
        
        # Precompute initial degrees
        degree = [0] * len(elements)
        for i, j, k in conflicts:
            degree[i] += 1
            degree[j] += 1
            degree[k] += 1
            
        # We need to update degrees as we remove elements.
        # This is expensive in Python if not careful.
        
        # Alternative: Simple Greedy
        # 1. Sort elements by degree descending.
        # 2. Iterate through sorted list.
        # 3. If element is still active, remove it if it's in any active conflict?
        #    No, that's not right.
        
        # Let's use the "Pick random conflict, remove best vertex" strategy with maintained counts.
        
        # counts[e] = number of active conflicts involving e
        counts = [0] * len(elements)
        for i, j, k in conflicts:
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
            
        removed = [False] * len(elements)
        
        # To avoid O(N^2) per step, we can just pick a random active conflict.
        # We need a list of active conflicts.
        active_conf_list = list(active_conflicts)
        
        while active_conf_list:
            # Pick random active conflict
            conf_idx = rng.choice(active_conf_list)
            i, j, k = conflicts[conf_idx]
            
            # Find the best candidate among i, j, k that is not removed
            best_candidate = -1
            best_score = -1
            
            for c in [i, j, k]:
                if not removed[c]:
                    # Score is current count of active conflicts
                    # Note: counts[c] includes conf_idx.
                    if counts[c] > best_score:
                        best_score = counts[c]
                        best_candidate = c
                        
            if best_candidate == -1:
                # All removed? Remove from active list
                active_conf_list.remove(conf_idx)
                continue
                
            # Remove best_candidate
            removed[best_candidate] = True
            
            # Update counts for all conflicts involving best_candidate
            # We need to know which conflicts involve best_candidate.
            # We can precompute list of conflicts per element.
            # Let's precompute elem_to_conf_indices
            
            # This approach requires precomputation.
            pass

    # Let's implement a cleaner version of the Greedy Hitting Set.
    
    def solve_sum_free_heuristic(elements, seed):
        n = len(elements)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Find all conflicts (triples)
        triples = []
        # To avoid duplicates and ensure efficiency, we iterate pairs
        for i in range(n):
            vi = elements[i]
            for j in range(i, n):
                vj = elements[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Sort indices to have a canonical representation if needed, but here we just need the set
                    triples.append((i, j, k))
                    
        if not triples:
            return elements[:]
            
        # Precompute which triples involve each element
        elem_to_triples = {i: [] for i in range(n)}
        for idx, (i, j, k) in enumerate(triples):
            elem_to_triples[i].append(idx)
            elem_to_triples[j].append(idx)
            elem_to_triples[k].append(idx)
            
        # State
        active_triples = set(range(len(triples)))
        removed_elem = [False] * n
        
        # Counts of active triples per element
        active_count = [0] * n
        for i, j, k in triples:
            active_count[i] += 1
            active_count[j] += 1
            active_count[k] += 1
            
        rng = random.Random(seed)
        
        # Greedy loop
        # To speed up, we can maintain a max-heap or just pick randomly from active triples
        # and select the best vertex from that triple.
        
        active_triple_list = list(active_triples)
        
        while active_triple_list:
            # Pick a random active triple
            t_idx = rng.choice(active_triple_list)
            i, j, k = triples[t_idx]
            
            # Candidates
            candidates = []
            for c in [i, j, k]:
                if not removed_elem[c]:
                    candidates.append(c)
                    
            if not candidates:
                # This triple is already resolved (all its elements removed)
                # Remove it from active list
                active_triple_list.remove(t_idx)
                continue
                
            # Choose the candidate with the highest active_count
            best_c = max(candidates, key=lambda x: active_count[x])
            
            # Remove best_c
            removed_elem[best_c] = True
            
            # Update active_count for all triples involving best_c
            for tr_idx in elem_to_triples[best_c]:
                if tr_idx in active_triples:
                    # This triple is now resolved if all its elements are removed?
                    # No, we just removed one. The triple is "broken" if ANY element is removed.
                    # So we remove the triple from active set.
                    active_triples.remove(tr_idx)
                    if tr_idx in active_triple_list:
                        active_triple_list.remove(tr_idx)
                        
                    # Decrease count for the OTHER elements in this triple
                    a, b, c_val = triples[tr_idx]
                    for e in [a, b, c_val]:
                        if e != best_c:
                            active_count[e] -= 1
                            
            # Note: We don't decrease active_count for best_c anymore as it's removed.
            
        # Result
        result = [elements[i] for i in range(n) if not removed_elem[i]]
        return result

    results = []
    # Use multiple seeds for the heuristic to find a better solution
    # The instance seed is fixed, but the heuristic seed can vary
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        # Run heuristic with a few different random seeds to pick the best
        best_res = []
        best_len = 0
        for h_seed in range(10): # 10 restarts
            res = solve_sum_free_heuristic(elems, seed + h_seed)
            if len(res) > best_len:
                best_len = len(res)
                best_res = res
        results.append(sorted(best_res))
        
    return results
# EVOLVE-BLOCK-END
