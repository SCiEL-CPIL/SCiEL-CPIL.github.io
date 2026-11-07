import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for O(1) lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute validity of all subsets (bitmasks) as clique or independent set
    # This is the key insight from the "max satisfiability over bitmask" strategy.
    # For n=14, 2^14 = 16384 subsets.
    is_clique_mask = [False] * (1 << n)
    is_indep_mask = [False] * (1 << n)
    
    # Empty set is both
    is_clique_mask[0] = True
    is_indep_mask[0] = True
    
    # Single vertices are both
    for v in range(n):
        mask = 1 << v
        is_clique_mask[mask] = True
        is_indep_mask[mask] = True
        
    # Compute for all larger masks using DP
    # A subset S is a clique if for any v in S, S\{v} is a clique and v is adjacent to all in S\{v}
    # A subset S is an independent set if for any v in S, S\{v} is an independent set and v is NOT adjacent to any in S\{v}
    
    for mask in range(1, 1 << n):
        if is_clique_mask[mask] or is_indep_mask[mask]:
            continue # Already computed (size 0 or 1)
            
        # Find a vertex in the mask
        # We can pick the lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique_mask[rest]:
            # Check if v is adjacent to all in rest
            # v is adjacent to all in rest iff (adj[v] & rest) == rest
            if (adj[v] & rest) == rest:
                is_clique_mask[mask] = True
        
        # Check independent
        if is_indep_mask[rest]:
            # Check if v is NOT adjacent to any in rest
            # v is not adjacent to any in rest iff (adj[v] & rest) == 0
            if (adj[v] & rest) == 0:
                is_indep_mask[mask] = True

    # Precompute list of valid masks (non-empty)
    # We will use these to build partitions.
    # However, generating all partitions is expensive.
    # Instead, we use the backtracking approach but with bitmask state.
    
    # Order vertices by descending degree for better pruning in backtracking
    degrees = [len(adj[v]) for v in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    start_time = time.time()
    deadline = start_time + 55.0 # 55s margin out of 90s
    
    # We try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # State for backtracking:
        # parts_masks: list of k integers (bitmasks of vertices in each part)
        # We assign vertices one by one in the sorted order.
        
        # To make the search faster, we can also precompute for each vertex v and each current part mask m,
        # whether v can be added to m as a clique or independent set.
        # But the dynamic check is fast enough with bitmasks.
        
        def dfs(idx, parts_masks):
            if time.time() > deadline:
                return None
            
            if idx == n:
                # All vertices assigned
                # Return the non-empty parts
                res = []
                for m in parts_masks:
                    if m:
                        # Convert mask to list of vertices
                        verts = []
                        temp_m = m
                        while temp_m:
                            lsb = temp_m & -temp_m
                            v = lsb.bit_length() - 1
                            verts.append(v)
                            temp_m ^= lsb
                        res.append(verts)
                return res
            
            v = vertices[idx]
            v_mask = 1 << v
            
            # Find first empty part index for symmetry breaking
            first_empty = -1
            for i in range(k):
                if parts_masks[i] == 0:
                    first_empty = i
                    break
            
            for i in range(k):
                # Symmetry breaking
                if parts_masks[i] == 0 and i != first_empty:
                    continue
                
                current_mask = parts_masks[i]
                
                # Try adding v as a clique extension
                new_mask_c = current_mask | v_mask
                if is_clique_mask[new_mask_c]:
                    parts_masks[i] = new_mask_c
                    res = dfs(idx + 1, parts_masks)
                    if res is not None:
                        return res
                    parts_masks[i] = current_mask
                
                # Try adding v as an independent set extension
                # Note: If current_mask is 0, both clique and indep are valid for single vertex.
                # We already tried clique. If current_mask > 0, we try indep.
                # If current_mask == 0, we already handled it in the clique branch (since single vertex is both).
                # But wait, if we add v to an empty part, it becomes a single vertex part.
                # The type (clique vs indep) doesn't matter for a single vertex.
                # So we only need to try one branch for empty parts?
                # Actually, the state `parts_masks` doesn't track the *type* of the part, only the set of vertices.
                # The validity is determined by the set itself.
                # So if `is_clique_mask[new_mask_c]` is true, we add it.
                # If `is_indep_mask[new_mask_c]` is true, we add it.
                # Since a set can be both (size 1), we just need to check if the new set is valid.
                
                if is_indep_mask[new_mask_c]:
                    parts_masks[i] = new_mask_c
                    res = dfs(idx + 1, parts_masks)
                    if res is not None:
                        return res
                    parts_masks[i] = current_mask
            
            return None
        
        init_masks = [0] * k
        res = dfs(0, init_masks)
        
        if res is not None:
            # Ensure types are correct
            final_parts = []
            for part in res:
                clean_part = [int(v) for v in part]
                if clean_part:
                    final_parts.append(clean_part)
            return final_parts

    # Fallback
    return [[i] for i in range(n)]
