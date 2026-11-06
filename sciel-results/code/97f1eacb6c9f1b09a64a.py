<<<<<<< SEARCH
    # Helper: Find Maximum Independent Set size in induced subgraph on vertices V_set
    # Simple Branch and Bound
    def max_independent_set_size(V_set):
        if not V_set:
            return 0
        # Greedy lower bound
        # Sort by degree
        # For simplicity, use a recursive search with pruning
        # Given small n, this should be fast enough
        
        V_list = list(V_set)
        # Heuristic: pick vertex with max degree first?
        # Or min degree?
        
        best = 0
        
        def dfs(remaining, count):
            nonlocal best
            if not remaining:
                if count > best:
                    best = count
                return
            if count + len(remaining) <= best:
                return
            
            # Pick a vertex
            # To prune effectively, pick a vertex with high degree in the subgraph
            # Let's just pick the first one for simplicity, or random?
            # Randomization helps with different runs
            v = random.choice(list(remaining))
            
            # Option 1: Include v
            # Neighbors of v in remaining must be excluded
            neighbors_in_rem = current_adj[v].intersection(remaining)
            new_remaining = remaining - neighbors_in_rem - {v}
            dfs(new_remaining, count + 1)
            
            # Option 2: Exclude v
            dfs(remaining - {v}, count)

        dfs(set(V_list), 0)
        return best
=======
    # Helper: Check if there is an independent set of size k in the induced subgraph defined by vertices_mask
    # Uses bitset branch and bound
    def has_independent_set_size_k(vertices_mask, adj_masks, n, k):
        if vertices_mask == 0:
            return False
        
        found = False
        
        def dfs(remaining_mask, count):
            nonlocal found
            if found:
                return
            if count == k:
                found = True
                return
            
            if remaining_mask == 0:
                return
            
            # Pruning: if count + popcount(remaining) < k, return
            if count + bin(remaining_mask).count('1') < k:
                return
            
            # Pick vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(remaining_mask & adj_masks[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_mask ^= lsb
            
            v = best_v
            v_bit = 1 << v
            
            # Option 1: Include v
            new_remaining_inc = remaining_mask & ~(adj_masks[v] | v_bit)
            dfs(new_remaining_inc, count + 1)
            
            if found:
                return
            
            # Option 2: Exclude v
            new_remaining_exc = remaining_mask & ~v_bit
            dfs(new_remaining_exc, count)

        dfs(vertices_mask, 0)
        return found
>>>>>>> REPLACE

