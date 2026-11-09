import time
import sys

def construct():
    # Seed data
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix/bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute clique and independent set properties for all subsets
    # is_clique[mask] is True if mask induces a clique
    # is_indep[mask] is True if mask induces an independent set
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    
    # 0 is both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    
    # Single vertices
    for v in range(n):
        mask = 1 << v
        is_clique[mask] = True
        is_indep[mask] = True
        
    # Compute for larger masks
    # A set is a clique if for every v in set, (set without v) is a clique and v is connected to all in (set without v)
    # A set is an independent set if for every v in set, (set without v) is an independent set and v is NOT connected to any in (set without v)
    
    for mask in range(1, size):
        if is_clique[mask] or is_indep[mask]:
            continue # Already computed? No, we need to compute for all.
            # Actually, we can compute iteratively.
        
    # Better approach:
    # For each mask, pick lowest set bit v.
    # rest = mask ^ (1 << v)
    # is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
    # is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)
    
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        # Find lowest set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)
        
    # Fallback greedy solution (same as seed)
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    best_parts = parts
    best_num_parts = len(parts)
    
    # Search for better solution
    start_time = time.time()
    deadline = start_time + 55 # 55s margin
    
    # DFS state
    # masks: list of k ints
    # v: current vertex index
    # k: number of parts allowed
    
    # We search for k from 1 to best_num_parts - 1
    # If we find a solution with k parts, it's better.
    
    def solve_k(k):
        # Returns list of masks if solution found, else None
        # Symmetry breaking: parts are ordered by creation time.
        # We only add to existing parts or create a new part at the end.
        
        masks = [0] * k
        # We don't need to store the full path if we just want to return one solution.
        # But we need to reconstruct it.
        # We can store the solution in a global variable or return it.
        # Since we are in a closure, we can use a nonlocal variable.
        
        solution = [None]
        
        def dfs(v, masks):
            if time.time() > deadline:
                return False
                
            if v == n:
                # Check if all parts are non-empty?
                # The symmetry breaking ensures we only create new parts if needed,
                # but we might have empty parts if k > actual used.
                # However, we are searching for exactly k parts?
                # No, we are searching for AT MOST k parts.
                # But the symmetry breaking "create new part at end" means
                # if we use fewer than k parts, the remaining masks will be 0.
                # A partition into 0 parts is invalid for a part.
                # So we must ensure that the number of non-zero masks is <= k.
                # And each non-zero mask is a valid part.
                # The DFS ensures validity of each mask as it grows.
                # So if we reach v==n, all masks are valid cliques or ind sets.
                # We just need to check if the number of non-empty parts is <= k.
                # Actually, the search space is restricted to using parts 0..k-1.
                # If some are empty, that's fine, it means we used fewer parts.
                # But we want to minimize the number of parts.
                # So if we find a solution with k=4, but it only uses 3 parts,
                # that would have been found in the k=3 search?
                # Not necessarily, because the search for k=3 restricts us to 3 parts.
                # So if we find a valid assignment using 3 parts during k=4 search,
                # it would also be a valid assignment for k=3.
                # So the first k that yields a solution is the minimum.
                
                # Extract solution
                parts_found = []
                for m in masks:
                    if m != 0:
                        verts = []
                        for i in range(n):
                            if m & (1 << i):
                                verts.append(i)
                        parts_found.append(verts)
                solution[0] = parts_found
                return True
                
            # Try adding v to existing parts
            # Determine number of non-empty parts so far
            # To optimize, we can just iterate 0 to k-1
            for i in range(k):
                # If part i is empty, we can only put v here if all previous parts are empty?
                # No, symmetry breaking:
                # If part i is empty, and we put v in it, it becomes non-empty.
                # This is only allowed if i is the first empty part.
                # i.e., parts 0..i-1 are non-empty, part i is empty.
                # This ensures we don't have "holes" in the part indices.
                
                if masks[i] == 0:
                    # Check if any part j < i is empty?
                    # If so, we should have put v in j first.
                    # So we only allow creating a new part at the first empty slot.
                    # If masks[i] == 0, we must ensure masks[j] != 0 for all j < i.
                    # Actually, if masks[j] == 0 for j < i, we should have tried that.
                    # So if we are here, it implies masks[0..i-1] are non-empty?
                    # Not necessarily, we might have skipped them?
                    # No, the loop goes 0 to k-1.
                    # If masks[0] was 0, we would have tried adding v to it in iteration 0.
                    # If that failed or led to dead end, we move to 1.
                    # But if masks[0] is 0, adding v to it creates part 0.
                    # Then masks[1] is still 0.
                    # So we can only add v to the FIRST empty part.
                    
                    # Check if this is the first empty part
                    is_first_empty = True
                    for j in range(i):
                        if masks[j] == 0:
                            is_first_empty = False
                            break
                            
                    if not is_first_empty:
                        continue
                        
                    # Try adding v to part i (which is 0)
                    new_mask = 1 << v
                    # Single vertex is always valid
                    masks[i] = new_mask
                    if dfs(v + 1, masks):
                        return True
                    masks[i] = 0
                else:
                    # Part i is non-empty
                    new_mask = masks[i] | (1 << v)
                    if is_clique[new_mask] or is_indep[new_mask]:
                        masks[i] = new_mask
                        if dfs(v + 1, masks):
                            return True
                        masks[i] = 0
                        
            return False
            
        dfs(0, masks)
        return solution[0]
        
    # Try to improve
    for k in range(1, best_num_parts):
        if time.time() > deadline:
            break
        sol = solve_k(k)
        if sol is not None:
            best_parts = sol
            best_num_parts = len(sol)
            break # Found minimum, stop
            
    return best_parts
