import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Build bitsets for fast clique search
    # For each node, store a list of bitmasks representing its non-neighbors (compatible nodes)
    # We use Python integers as arbitrary-precision bitsets
    compat_bits = [0] * N
    for i in range(N):
        bits = 0
        for j in range(N):
            if i != j and not Conflict[i, j]:
                bits |= (1 << j)
        compat_bits[i] = bits
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees: subtract conflicts with removed nodes
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices, max_iters=20):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find neighbors of i not in selected_set
                nb_i = np.where(Conflict[i] == 1)[0]
                valid_js = [j for j in nb_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each j, check if it conflicts with any k in S \ {i}
                sel_arr = np.array([k for k in selected_set if k != i])
                if len(sel_arr) == 0:
                    compatible_with_rest = valid_js
                else:
                    # Conflict[j, sel_arr] for j in valid_js
                    # Shape: (len(valid_js), len(sel_arr))
                    conflicts_matrix = Conflict[np.array(valid_js)[:, None], sel_arr[None, :]]
                    # j is compatible if no conflicts
                    compat_mask = (conflicts_matrix.sum(axis=1) == 0)
                    compatible_with_rest = [valid_js[j] for j in range(len(valid_js)) if compat_mask[j]]
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a compatible pair in compatible_with_rest
                cwr = compatible_with_rest
                found_pair = False
                for p in range(len(cwr)):
                    if found_pair:
                        break
                    j1 = cwr[p]
                    for q in range(p + 1, len(cwr)):
                        j2 = cwr[q]
                        if not Conflict[j1, j2]:
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    break
                
        return list(selected_set)

    def max_clique_bnb(start_node, time_limit=1.0):
        """Branch and bound for maximum clique using bitsets."""
        if time.time() - start_node > time_limit:
            return []
        
        # Greedy coloring to get upper bound
        # We'll use a simple sequential coloring on the candidate set
        
        def color_bound(candidates):
            """Return (color_count, color_of) for greedy coloring of candidates."""
            if not candidates:
                return 0, {}
            
            # Order by degree in the subgraph induced by candidates
            cand_set = set(candidates)
            degrees = {}
            for v in candidates:
                # Count neighbors within candidates
                deg = 0
                for u in candidates:
                    if v != u and (compat_bits[v] >> u) & 1:
                        deg += 1
                degrees[v] = deg
            
            # Sort by degree descending
            sorted_cands = sorted(candidates, key=lambda x: -degrees[x])
            
            colors = {}
            color_of = {}
            next_color = 0
            used_colors = []
            
            for v in sorted_cands:
                # Find the smallest color not used by neighbors of v in the subgraph
                neighbor_colors = set()
                for u in sorted_cands:
                    if u != v and (compat_bits[v] >> u) & 1:
                        if u in color_of:
                            neighbor_colors.add(color_of[u])
                
                # Assign smallest available color
                c = 0
                while c in neighbor_colors:
                    c += 1
                color_of[v] = c
                next_color = max(next_color, c + 1)
            
            return next_color, color_of
        
        best_clique = []
        
        def branch(candidates, clique, remaining_time):
            nonlocal best_clique
            
            if time.time() - start_node > remaining_time:
                return
            
            if not candidates:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
                return
            
            # Get upper bound from coloring
            color_count, color_of = color_bound(candidates)
            
            # If current clique size + upper bound <= best, prune
            if len(clique) + color_count <= len(best_clique):
                return
            
            # Order candidates by color (ascending) for better pruning
            sorted_cands = sorted(candidates, key=lambda x: color_of.get(x, 0))
            
            for v in sorted_cands:
                if time.time() - start_node > remaining_time:
                    return
                
                # If color of v + len(clique) <= best, no point continuing
                if color_of.get(v, 0) + len(clique) <= len(best_clique):
                    break
                
                new_clique = clique + [v]
                # New candidates: those in candidates that are compatible with v and come after v in order
                new_cands = [u for u in sorted_cands if (compat_bits[v] >> u) & 1]
                
                branch(new_cands, new_clique, remaining_time)
        
        # Start with an initial greedy solution
        greedy_clique = []
        remaining = list(range(N))
        while remaining:
            if time.time() - start_node > time_limit:
                break
            # Pick node with max degree in remaining
            best_v = max(remaining, key=lambda v: sum(1 for u in remaining if u != v and (compat_bits[v] >> u) & 1))
            greedy_clique.append(best_v)
            remaining = [u for u in remaining if (compat_bits[best_v] >> u) & 1]
        
        best_clique = greedy_clique
        
        # Run branch and bound
        branch(list(range(N)), [], time_limit - (time.time() - start_node))
        
        return best_clique

    # First, run greedy + local search to get a good initial solution
    for s in range(100):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
    
    # Now try maximum clique search with time budget
    if time.time() < deadline - 5:
        # Try a few different starting points for B&B
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            # Create a random ordering for the B&B
            rng_bb = np.random.default_rng(attempt * 12345 + 67890)
            perm = rng_bb.permutation(N)
            
            # Run B&B with limited time
            bnb_start = time.time()
            bnb_time_left = min(10.0, deadline - time.time() - 1.0)
            if bnb_time_left <= 0:
                break
            
            # For B&B, we need to work with the permuted indices
            # We'll create a mapping
            result = max_clique_bnb(0, bnb_time_left)
            
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
    
    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
