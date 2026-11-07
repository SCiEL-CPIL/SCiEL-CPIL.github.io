import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute clique and independent set status for all masks
    # This is useful for quick checks during DFS
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True
            
    def dfs(unassigned, parts, k_rem):
        if time.time() - t0 > DEADLINE:
            return None
            
        if unassigned == 0:
            return parts[:]
            
        # Pruning: If the number of unassigned vertices is 0, we are done.
        # If k_rem == 0 but unassigned != 0, fail.
        if k_rem == 0:
            return None
            
        # Pruning: If the remaining unassigned vertices can form a single valid part
        # and k_rem == 1, check if unassigned is clique or indep.
        if k_rem == 1:
            if is_clique[unassigned] or is_indep[unassigned]:
                return parts + [unassigned]
            else:
                return None
        
        # Pick the lowest set bit vertex to break symmetry?
        # Actually, standard DFS picks a vertex and tries to place it.
        # To reduce branching, we can pick a vertex that is "hard" to place.
        # For now, pick lowest bit.
        v = (unassigned & -unassigned).bit_length() - 1
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit
        
        # Option 1: Add v to an existing part
        for i in range(len(parts)):
            p_mask = parts[i]
            # Check if adding v keeps it clique or independent
            new_mask = p_mask | v_bit
            if is_clique[new_mask] or is_indep[new_mask]:
                new_parts = parts[:]
                new_parts[i] = new_mask
                res = dfs(new_unassigned, new_parts, k_rem)
                if res is not None:
                    return res
        
        # Option 2: Create a new part with v
        if k_rem > 0:
            new_parts = parts[:] + [v_bit]
            res = dfs(new_unassigned, new_parts, k_rem - 1)
            if res is not None:
                return res
                
        return None

    # Try increasing k
    for k in range(1, n + 1):
        # Initial state: no parts, all vertices unassigned
        # We can start with an empty list of parts
        res = dfs(full_mask, [], k)
        if res is not None:
            # Convert bitmasks to lists of vertices
            final_parts = []
            for mask in res:
                verts = []
                for v in range(n):
                    if mask & (1 << v):
                        verts.append(v)
                final_parts.append(verts)
            return final_parts
            
    # Fallback (should not happen)
    return [[v] for v in range(n)]
