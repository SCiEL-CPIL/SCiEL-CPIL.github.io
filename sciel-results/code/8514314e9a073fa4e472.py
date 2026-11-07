"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = t0 + 55.0  # safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # non_adj[v] = bitmask of vertices not adjacent to v (excluding v itself)
    non_adj = [0] * n
    for v in range(n):
        # All vertices except v and v's neighbors
        non_adj[v] = ((1 << n) - 1) ^ (1 << v) ^ adj[v]
    
    # Check if a mask is a clique
    def is_clique(mask):
        # For each vertex v in mask, all other vertices in mask must be neighbors
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            others = mask ^ lsb
            if (adj[v] & others) != others:
                return False
        return True
    
    # Check if a mask is an independent set
    def is_indep(mask):
        # For each vertex v in mask, no other vertex in mask should be a neighbor
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            others = mask ^ lsb
            if (adj[v] & others) != 0:
                return False
        return True
    
    def is_valid_part(mask):
        return is_clique(mask) or is_indep(mask)
    
    # Greedy fallback
    def greedy():
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Backtracking to find partition with exactly K parts
    def solve_k(K):
        # State: list of K bitmasks (initially 0), unassigned mask
        # We assign vertices one at a time to one of the K parts
        # Parts are unlabeled, so we use canonical ordering to avoid symmetry
        
        # Precompute: for each possible part mask, whether it's valid
        # But masks can be up to 2^14 = 16384, which is fine
        
        # We'll use DFS: pick an unassigned vertex, try placing it in each part
        # To reduce symmetry, parts are kept in sorted order (by bitmask value)
        # and we only allow placing vertex in parts that come after or at the last non-empty part
        
        def dfs(parts, unassigned):
            if time.time() > DEADLINE:
                return None
            
            if unassigned == 0:
                # All vertices assigned, verify all parts are non-empty and valid
                # (they should be valid by construction, but double-check)
                for p in parts:
                    if p == 0:
                        continue
                    if not is_valid_part(p):
                        return None
                # Convert to lists
                result = []
                for p in parts:
                    if p != 0:
                        lst = []
                        m = p
                        while m:
                            lsb = m & (-m)
                            lst.append(lsb.bit_length() - 1)
                            m ^= lsb
                        lst.sort()
                        result.append(lst)
                return result
            
            # Pick the lowest unassigned vertex
            lsb = unassigned & (-unassigned)
            v = lsb.bit_length() - 1
            new_unassigned = unassigned ^ lsb
            v_bit = 1 << v
            
            # Try placing v in each part
            # To reduce symmetry: only consider parts that are "active" 
            # (non-empty or the first empty part after all non-empty ones)
            # Actually, simpler: try all K parts but break early if we've tried 
            # an equivalent assignment
            
            tried_masks = set()
            for i in range(K):
                # Symmetry breaking: if this part has the same mask as a previously tried part, skip
                if parts[i] in tried_masks:
                    continue
                tried_masks.add(parts[i])
                
                # Also, if part i is empty and there's a later empty part, skip 
                # (only use the first empty part)
                if parts[i] == 0:
                    for j in range(i+1, K):
                        if parts[j] == 0:
                            break
                    else:
                        pass  # This is the last empty part, OK
                    # Actually, just skip if there's any earlier non-empty... no.
                    # Let me just allow all but use the tried_masks dedup
                    # The tried_masks should handle identical empty parts (all 0)
                    # So after first 0 is tried, subsequent 0s are skipped
                    pass
                
                new_part = parts[i] | v_bit
                if not is_valid_part(new_part):
                    continue
                
                # Recurse
                new_parts = list(parts)
                new_parts[i] = new_part
                result = dfs(new_parts, new_unassigned)
                if result is not None:
                    return result
            
            return None
        
        # Initial state: K empty parts
        initial_parts = [0] * K
        return dfs(initial_parts, (1 << n) - 1)
    
    # Try increasing K
    best_result = None
    for K in range(2, n + 1):
        if time.time() > DEADLINE:
            break
        result = solve_k(K)
        if result is not None:
            # Verify: all vertices present exactly once
            all_vertices = []
            for part in result:
                all_vertices.extend(part)
            if sorted(all_vertices) == list(range(n)):
                # Verify each part is non-empty and valid
                valid = True
                for part in result:
                    if len(part) == 0:
                        valid = False
                        break
                    mask = 0
                    for v in part:
                        mask |= (1 << v)
                    if not is_valid_part(mask):
                        valid = False
                        break
                if valid:
                    best_result = result
                    break
    
    if best_result is None:
        best_result = greedy()
    
    # Final verification
    all_vertices = []
    for part in best_result:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(n)):
        best_result = greedy()
    
    # Ensure types are correct
    result = []
    for part in best_result:
        result.append([int(v) for v in part])
    return result
# EVOLVE-BLOCK-END
