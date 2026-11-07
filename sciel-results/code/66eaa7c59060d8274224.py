import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    # Non-adjacency mask
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = all_mask & ~(adj[v] | (1 << v))

    deadline = time.time() + 55
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
    
    def solve_k(k):
        """
        Try to partition vertices into k parts.
        Parts are dynamically typed (Clique or Independent Set).
        """
        # State:
        # masks[i]: bitmask of vertices in part i
        # types[i]: 0=Unknown (size 0 or 1), 1=Clique, 2=Independent
        # allows[i]: bitmask of vertices that can be added to part i
        masks = [0] * k
        types = [0] * k
        allows = [all_mask] * k
        
        # Track which vertices are assigned
        assigned_mask = 0
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            v_bit = 1 << v
            
            # Find the first empty part index to enforce symmetry breaking
            # We only place v in the first available empty part, or in non-empty parts.
            # Actually, we iterate all parts, but if part i is empty, we skip it if any part j < i is empty.
            
            first_empty = -1
            for i in range(k):
                if masks[i] == 0:
                    first_empty = i
                    break
            
            for i in range(k):
                # Symmetry breaking:
                # If this part is empty, and there is an earlier empty part, skip this one.
                if masks[i] == 0:
                    if i != first_empty:
                        continue
                
                # Check if v is allowed in part i
                if not (allows[i] & v_bit):
                    continue
                
                # Save state
                old_mask = masks[i]
                old_type = types[i]
                old_allow = allows[i]
                
                # Update state
                masks[i] |= v_bit
                
                if old_mask == 0:
                    # Part was empty, now has size 1
                    # Type remains 0 (Unknown)
                    # Allow remains all_mask? 
                    # Yes, because any next vertex will determine the type.
                    # However, we can restrict allows[i] to all_mask for now.
                    types[i] = 0
                    allows[i] = all_mask
                else:
                    # Part had at least 1 vertex
                    if types[i] == 0:
                        # Part had size 1
                        # Determine new type based on v and the existing vertex
                        # Find the existing vertex u
                        # old_mask has exactly 1 bit set
                        u_bit = old_mask
                        # Check if v is adjacent to u
                        if adj[v] & u_bit:
                            types[i] = 1 # Clique
                            # New allow: neighbors of v AND neighbors of u
                            # Since u is the only one, allow = adj[v] & adj[u]
                            # But we don't have adj[u] directly in a list? We do.
                            # We need to identify u.
                            # u = (u_bit.bit_length() - 1)
                            u = u_bit.bit_length() - 1
                            allows[i] = adj[v] & adj[u]
                        else:
                            types[i] = 2 # Independent
                            u = u_bit.bit_length() - 1
                            allows[i] = non_adj[v] & non_adj[u]
                    elif types[i] == 1:
                        # Part is Clique
                        types[i] = 1
                        allows[i] = allows[i] & adj[v]
                    else:
                        # Part is Independent
                        types[i] = 2
                        allows[i] = allows[i] & non_adj[v]
                
                if backtrack(idx + 1):
                    return True
                
                # Restore state
                masks[i] = old_mask
                types[i] = old_type
                allows[i] = old_allow
            
            return False
        
        if backtrack(0):
            # Reconstruct parts
            parts = []
            for i in range(k):
                if masks[i]:
                    part_verts = []
                    m = masks[i]
                    for v in range(n):
                        if m & (1 << v):
                            part_verts.append(v)
                    parts.append(part_verts)
            return parts
        return None

    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        res = solve_k(k)
        if res:
            # Ensure return type is list of lists of ints
            return res
        
    # Fallback
    return [[v] for v in range(n)]
