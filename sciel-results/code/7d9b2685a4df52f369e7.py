import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Non-adjacency bitmask (excluding self)
    non_adj = [0] * n
    for u in range(n):
        # All vertices except u
        mask = (1 << n) - 1
        mask ^= (1 << u)
        # Non-adjacent are those in mask but not in adj[u]
        non_adj[u] = mask & ~adj[u]

    # Order vertices by degree descending
    deg = [bin(adj[u]).count('1') for u in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: -deg[x])
    
    # Map original vertex to index in sorted list? 
    # Actually, we just iterate through the sorted list to pick the next unassigned vertex.
    
    def is_clique(mask):
        # Check if all pairs in mask are adjacent
        # For each vertex in mask, check if its neighbors include all other vertices in mask
        m = mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            # Check if adj[v] contains (mask without v)
            if (adj[v] & (mask ^ v_bit)) != (mask ^ v_bit):
                return False
        return True

    def is_independent(mask):
        # Check if no pairs in mask are adjacent
        m = mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            # Check if non_adj[v] contains (mask without v)
            if (non_adj[v] & (mask ^ v_bit)) != (mask ^ v_bit):
                return False
        return True

    # We will store parts as a list of (mask, type)
    # type: 0 = Flexible (size 1), 1 = Clique, 2 = Independent
    
    best_parts = None
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    # Try k from 1 to n
    for k in range(1, n + 1):
        parts = []
        
        def dfs(idx, assigned_mask):
            nonlocal best_parts
            if time.time() - start_time > timeout:
                return False
            
            if assigned_mask == (1 << n) - 1:
                # Found a solution
                # Convert parts to lists of vertex indices
                res = []
                for mask, _ in parts:
                    verts = []
                    m = mask
                    while m:
                        v_bit = m & -m
                        v = v_bit.bit_length() - 1
                        verts.append(v)
                        m ^= v_bit
                    res.append(verts)
                best_parts = res
                return True
            
            # Find next unassigned vertex in sorted order
            v = -1
            for u in vertices:
                if not (assigned_mask & (1 << u)):
                    v = u
                    break
            
            if v == -1:
                return False # Should not happen if assigned_mask check passed

            v_bit = 1 << v
            
            # Try adding v to existing parts
            for i in range(len(parts)):
                mask, ptype = parts[i]
                
                can_add = False
                new_ptype = ptype
                
                if ptype == 0: # Flexible (size 1)
                    # v can always be added
                    can_add = True
                    # Determine new type
                    if adj[v] & mask:
                        new_ptype = 1 # Clique
                    else:
                        new_ptype = 2 # Independent
                elif ptype == 1: # Clique
                    # v must be adjacent to all in mask
                    if (adj[v] & mask) == mask:
                        can_add = True
                        new_ptype = 1
                elif ptype == 2: # Independent
                    # v must be non-adjacent to all in mask
                    if (non_adj[v] & mask) == mask:
                        can_add = True
                        new_ptype = 2
                
                if can_add:
                    parts[i] = (mask | v_bit, new_ptype)
                    if dfs(idx + 1, assigned_mask | v_bit):
                        return True
                    parts[i] = (mask, ptype) # Backtrack
            
            # Try creating a new part
            if len(parts) < k:
                parts.append((v_bit, 0))
                if dfs(idx + 1, assigned_mask | v_bit):
                    return True
                parts.pop()
                
            return False

        if dfs(0, 0):
            if best_parts:
                return best_parts
        # If timeout or no solution for this k, try next k
        if time.time() - start_time > timeout:
            break

    # Fallback: Greedy solution if search failed (should not happen for n=14)
    # But we must return a valid partition.
    # The seed program's greedy is valid. We can just return a valid partition.
    # Actually, if the search fails, it might be due to timeout.
    # Let's run a quick greedy to ensure we return *something* valid.
    
    # Greedy fallback
    parts_greedy = []
    es = {tuple(e) for e in edges}
    for v in range(n):
        placed = False
        for part in parts_greedy:
            # Check if v can be added to part
            # Determine if part is clique or independent
            # This is slow, but only used as fallback.
            is_c = True
            is_i = True
            for i, u in enumerate(part):
                for w in part[i+1:]:
                    if (u, w) in es:
                        is_i = False
                    else:
                        is_c = False
            if is_c or is_i:
                # Check compatibility with v
                ok = True
                if is_c and not is_i:
                    for u in part:
                        if (u, v) not in es and (v, u) not in es:
                            ok = False
                            break
                elif is_i and not is_c:
                    for u in part:
                        if (u, v) in es or (v, u) in es:
                            ok = False
                            break
                else: # Flexible (size 1)
                    ok = True
                
                if ok:
                    part.append(v)
                    placed = True
                    break
        if not placed:
            parts_greedy.append([v])
            
    return parts_greedy
