"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute validity of all subsets
    # valid[mask] is True if mask is a clique or an independent set
    valid = [False] * (1 << n)
    max_part_size = 1
    valid[0] = True # Empty set is valid, though we don't use empty parts
    
    for mask in range(1, 1 << n):
        is_clique = True
        is_ind = True
        # Iterate through vertices in mask
        m = mask
        while m:
            l = m & -m
            v = l.bit_length() - 1
            neighbors_in_mask = adj_mask[v] & mask
            # For clique: neighbors in mask must be all other vertices in mask
            if neighbors_in_mask != (mask ^ l):
                is_clique = False
            # For independent set: no neighbors in mask
            if neighbors_in_mask != 0:
                is_ind = False
            if not is_clique and not is_ind:
                break
            m ^= l
            
        if is_clique or is_ind:
            valid[mask] = True
            size = mask.bit_count()
            if size > max_part_size:
                max_part_size = size
                
    full_mask = (1 << n) - 1
    best_parts = None
    
    # Search for minimum k
    # k ranges from 1 to n
    for k in range(1, n + 1):
        parts = []
        
        def dfs(remaining, parts, k):
            if remaining == 0:
                return True
            
            # Pruning: if we have too many parts
            if len(parts) > k:
                return False
                
            # Pruning: lower bound on remaining parts needed
            rem_count = remaining.bit_count()
            import math
            min_parts_needed = math.ceil(rem_count / max_part_size)
            if len(parts) + min_parts_needed > k:
                return False
                
            # Select vertex with highest degree in the induced subgraph of 'remaining'
            # This helps prune faster as highly connected vertices are harder to place in independent sets
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                l = temp & -temp
                v = l.bit_length() - 1
                deg = (adj_mask[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= l
            
            v_mask = 1 << best_v
            new_remaining = remaining ^ v_mask
            
            # Try adding to existing parts
            for i in range(len(parts)):
                new_part = parts[i] | v_mask
                if valid[new_part]:
                    parts[i] = new_part
                    if dfs(new_remaining, parts, k):
                        return True
                    parts[i] ^= v_mask # Backtrack
                    
            # Try starting a new part
            if len(parts) < k:
                parts.append(v_mask)
                if dfs(new_remaining, parts, k):
                    return True
                parts.pop()
                
            return False
            
        if dfs(full_mask, parts, k):
            # Convert bitmasks to lists of integers
            result = []
            for p in parts:
                verts = []
                m = p
                while m:
                    l = m & -m
                    v = l.bit_length() - 1
                    verts.append(v)
                    m ^= l
                result.append(verts)
            return result
            
    # Fallback: should not happen, but return a trivial partition
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
