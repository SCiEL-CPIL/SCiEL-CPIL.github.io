"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Generate all maximal cliques
    # Bron-Kerbosch algorithm
    def expand_clique(r, p, x):
        if not p and not x:
            yield r
        # Choose pivot u from p union x to minimize branching
        # For simplicity and speed on n=14, we can just iterate
        # But let's use a simple recursive generation
        # Actually, for n=14, iterating all subsets and checking is 2^14=16384, which is very fast.
        # Let's stick to the simple method but only keep MAXIMAL ones to reduce candidate list.
        
    # Simple method: iterate all masks, check if clique, check if maximal.
    # A clique C is maximal if no vertex v outside C is adjacent to all vertices in C.
    
    max_cliques = []
    max_indep_sets = []
    
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        # Check if independent set
        is_indep = True
        
        # Get vertices
        verts = [v for v in range(n) if mask & (1 << v)]
        
        if not verts:
            continue
            
        # Check clique
        for v in verts:
            # v must be adjacent to all other vertices in verts
            others_mask = mask & ~(1 << v)
            if (adj[v] & others_mask) != others_mask:
                is_clique = False
                break
        
        if is_clique:
            # Check if maximal
            # A clique is maximal if there is no vertex u not in mask such that u is adjacent to all vertices in mask
            is_maximal = True
            for u in range(n):
                if not (mask & (1 << u)):
                    # u is not in clique
                    # check if u is adjacent to all v in mask
                    if (adj[u] & mask) == mask:
                        is_maximal = False
                        break
            if is_maximal:
                max_cliques.append(mask)
                
        # Check independent set
        for v in verts:
            # v must NOT be adjacent to any other vertex in verts
            others_mask = mask & ~(1 << v)
            if (adj[v] & others_mask) != 0:
                is_indep = False
                break
        
        if is_indep:
            # Check if maximal
            # An independent set is maximal if there is no vertex u not in mask such that u is NOT adjacent to any vertex in mask
            is_maximal = True
            for u in range(n):
                if not (mask & (1 << u)):
                    # u is not in set
                    # check if u is non-adjacent to all v in mask
                    # i.e., adj[u] & mask == 0
                    if (adj[u] & mask) == 0:
                        is_maximal = False
                        break
            if is_maximal:
                max_indep_sets.append(mask)
                
    # Combine candidates
    # We want to cover all vertices with disjoint sets from this list.
    # Note: Single vertices are maximal cliques and maximal independent sets.
    candidates = list(set(max_cliques + max_indep_sets))
    
    # Sort candidates by size descending to find large blocks first
    candidates.sort(key=lambda m: -bin(m).count('1'))
    
    # Precompute for each vertex, which candidates contain it
    # candidates_by_vertex[v] = list of indices in candidates containing v
    candidates_by_vertex = [[] for _ in range(n)]
    for i, c in enumerate(candidates):
        for v in range(n):
            if c & (1 << v):
                candidates_by_vertex[v].append(i)
                
    full_mask = (1 << n) - 1
    best_count = n + 1
    best_solution = None
    
    # Backtracking
    # To speed up, we can use a heuristic to pick the vertex:
    # Pick the vertex in remaining_mask that has the fewest valid candidates available.
    # This is the "most constrained variable" heuristic.
    
    def backtrack(remaining_mask, count, path):
        nonlocal best_count, best_solution
        
        if remaining_mask == 0:
            if count < best_count:
                best_count = count
                best_solution = path[:]
            return
            
        if count >= best_count:
            return
            
        # Heuristic: pick vertex v in remaining_mask with minimum number of valid candidates
        # Valid candidate: c is in candidates, v in c, and c subset of remaining_mask
        
        min_v = -1
        min_options = float('inf')
        
        # Iterate through vertices in remaining_mask
        temp_mask = remaining_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            # Count valid candidates for v
            # We can just count them on the fly or precompute?
            # Precomputing "subset of remaining" is hard as remaining changes.
            # So we count in loop.
            valid_opts = 0
            for idx in candidates_by_vertex[v]:
                c = candidates[idx]
                if (c & ~remaining_mask) == 0:
                    valid_opts += 1
                    # Early stop if we already have more than min_options
                    if valid_opts >= min_options:
                        break
                        
            if valid_opts < min_options:
                min_options = valid_opts
                min_v = v
                
                # If 0 options, dead end (should not happen if candidates cover all vertices)
                if min_options == 0:
                    break
                    
        if min_v == -1:
            return
            
        if min_options == 0:
            return
            
        # Try all valid candidates for min_v
        # Sort them by size descending?
        valid_candidates_for_v = []
        for idx in candidates_by_vertex[min_v]:
            c = candidates[idx]
            if (c & ~remaining_mask) == 0:
                valid_candidates_for_v.append((idx, c))
                
        # Sort by size descending
        valid_candidates_for_v.sort(key=lambda x: -bin(x[1]).count('1'))
        
        for idx, c in valid_candidates_for_v:
            path.append(c)
            backtrack(remaining_mask & ~c, count + 1, path)
            path.pop()
            
            # Optional: if we found a solution with count == lower_bound, we could stop
            # Lower bound: max(1, ceil(popcount(remaining) / max_candidate_size))
            # But max_candidate_size varies.
            # Simple bound: 1 if remaining != 0.
            # If count + 1 >= best_count, the loop above handles pruning.
            
    backtrack(full_mask, 0, [])
    
    if best_solution:
        # Convert masks to lists of vertices
        result = []
        for mask in best_solution:
            verts = [v for v in range(n) if mask & (1 << v)]
            result.append(verts)
        return result
    else:
        # Fallback
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
