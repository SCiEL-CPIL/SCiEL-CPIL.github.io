"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute validity of all subsets
    # valid[mask] is True if mask is a clique or an independent set
    size = 1 << n
    valid = [False] * size
    valid[0] = True
    
    # is_clique and is_indep
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Clique check
        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
                
        # Independent set check
        if is_indep[rest]:
            if (adj[v] & rest) == 0:
                is_indep[mask] = True
                
        if is_clique[mask] or is_indep[mask]:
            valid[mask] = True

    # IDDFS
    # parts is a list of bitmasks
    def dfs(v_idx, parts):
        if v_idx == n:
            return True
            
        # If we have used k parts, we can't make a new one
        # We try to add v_idx to existing parts first
        for i in range(len(parts)):
            new_part = parts[i] | (1 << v_idx)
            if valid[new_part]:
                # Update parts
                parts[i] = new_part
                if dfs(v_idx + 1, parts):
                    return True
                parts[i] = parts[i] ^ (1 << v_idx) # Revert
                
        # Try to start a new part
        if len(parts) < k:
            parts.append(1 << v_idx)
            if dfs(v_idx + 1, parts):
                return True
            parts.pop()
            
        return False

    # Find minimal k
    for k in range(1, n + 1):
        # Optional: Lower bound check
        # Chromatic number lower bound?
        # Clique number lower bound?
        # For now, just try.
        if dfs(0, []):
            # Reconstruct partition
            # We need to run it again to get the parts, or store them
            # Since dfs returns True, we need to capture the final state
            # Let's modify dfs to return the parts if successful
            pass
            
    # Redo with capture
    def dfs_capture(v_idx, parts):
        if v_idx == n:
            return True, parts[:]
        
        for i in range(len(parts)):
            new_part = parts[i] | (1 << v_idx)
            if valid[new_part]:
                parts[i] = new_part
                res, final_parts = dfs_capture(v_idx + 1, parts)
                if res:
                    return True, final_parts
                parts[i] = parts[i] ^ (1 << v_idx)
                
        if len(parts) < k:
            parts.append(1 << v_idx)
            res, final_parts = dfs_capture(v_idx + 1, parts)
            if res:
                return True, final_parts
            parts.pop()
            
        return False, None

    result_parts = None
    for k in range(1, n + 1):
        res, result_parts = dfs_capture(0, [])
        if res:
            break
            
    # Convert to list of lists
    partition = []
    for mask in result_parts:
        part_vertices = []
        for i in range(n):
            if mask & (1 << i):
                part_vertices.append(i)
        partition.append(part_vertices)
        
    return partition

# EVOLVE-BLOCK-END
