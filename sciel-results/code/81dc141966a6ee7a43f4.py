"""Rewritten program using bitmask backtracking for cochromatic partition."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj_mask = [0] * N
non_adj_mask = [0] * N
full_mask = (1 << N) - 1

for i in range(N):
    non_adj_mask[i] = full_mask & ~(1 << i)

for u, v in P['edges']:
    adj_mask[u] |= (1 << v)
    adj_mask[v] |= (1 << u)
    non_adj_mask[u] &= ~(1 << v)
    non_adj_mask[v] &= ~(1 << u)

def get_vertices(mask):
    verts = []
    while mask:
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        verts.append(i)
        mask ^= lsb
    return verts

def construct():
    best = N
    best_parts = None
    parts = [] # List of (type, mask). type 0: clique, 1: independent
    
    def dfs(rem):
        nonlocal best, best_parts
        
        if len(parts) >= best:
            return
        
        if rem == 0:
            if len(parts) < best:
                best = len(parts)
                best_parts = [p for p in parts]
            return
        
        # Select vertex with max degree in remaining subgraph
        v = -1
        max_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            u = lsb.bit_length() - 1
            deg = (adj_mask[u] & rem).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp_rem ^= lsb
        
        if v == -1:
            return
            
        v_bit = 1 << v
        new_rem = rem & ~v_bit
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            p_type, p_mask = parts[i]
            if p_type == 0: # Clique
                if (p_mask & adj_mask[v]) == p_mask:
                    parts[i] = (0, p_mask | v_bit)
                    dfs(new_rem)
                    parts[i] = (0, p_mask)
            else: # Independent
                if (p_mask & non_adj_mask[v]) == p_mask:
                    parts[i] = (1, p_mask | v_bit)
                    dfs(new_rem)
                    parts[i] = (1, p_mask)
        
        # Try adding v to a new part
        # Symmetry breaking: 
        # We only create a new Clique part if it's the first Clique part, 
        # or if we allow it. To reduce symmetry, we can enforce that 
        # new parts are created in a specific order relative to existing ones.
        # A simple heuristic: Only create a new part of type T if no existing part of type T 
        # is "smaller" in some sense? 
        # Let's just try both, but prune if len(parts) + 1 >= best.
        
        if len(parts) + 1 < best:
            # New Clique
            parts.append((0, v_bit))
            dfs(new_rem)
            parts.pop()
            
            # New Independent
            parts.append((1, v_bit))
            dfs(new_rem)
            parts.pop()
            
    dfs(full_mask)
    
    if best_parts is not None:
        result = []
        for p_type, p_mask in best_parts:
            verts = get_vertices(p_mask)
            result.append(verts)
        return result
    else:
        return [[i] for i in range(N)]

# EVOLVE-BLOCK-END
