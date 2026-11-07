"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    adj = [set() for _ in range(n)]
    for u, v in es:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build adjacency bitmasks for fast clique/independent checks
    adj_mask = [0] * n
    for u, v in es:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def is_clique_mask(mask):
        """Check if vertices in mask form a clique using bitmasks."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            # All other vertices in mask must be adjacent to v
            others = mask ^ low
            if others & ~adj_mask[v]:
                return False
            m ^= low
        return True
    
    def is_independent_mask(mask):
        """Check if vertices in mask form an independent set using bitmasks."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            # No other vertex in mask should be adjacent to v
            others = mask ^ low
            if others & adj_mask[v]:
                return False
            m ^= low
        return True
    
    def valid_add_mask(part_mask, v):
        """Check if vertex v can be added to part with bitmask part_mask."""
        if part_mask == 0:
            return True
        # Check clique: all vertices in part must be adjacent to v
        if (part_mask & ~adj_mask[v]) == 0:
            # Also need part itself to be a clique (but we maintain invariant)
            return True
        # Check independent: no vertex in part adjacent to v
        if (part_mask & adj_mask[v]) == 0:
            return True
        return False
    
    deadline = time.time() + 75  # Safety margin: 75s of 90s limit
    
    # Try multiple orderings
    orderings = [list(range(n))]
    deg_order = sorted(range(n), key=lambda x: -len(adj[x]))
    orderings.append(deg_order)
    
    best_result = None
    
    for k in range(1, n+1):
        if time.time() > deadline:
            break
        for order in orderings:
            if time.time() > deadline:
                break
            
            # Use bitmask representation for parts
            parts_mask = [0] * k
            parts_list = [[] for _ in range(k)]  # Track vertex lists for return
            
            def bt(idx):
                if time.time() > deadline:
                    return None
                if idx == len(order):
                    # Verify all parts are non-empty and valid
                    for i in range(k):
                        if parts_mask[i] == 0:
                            return None
                        if not (is_clique_mask(parts_mask[i]) or is_independent_mask(parts_mask[i])):
                            return None
                    return [list(p) for p in parts_list]
                
                v = order[idx]
                v_bit = 1 << v
                seen_empty = False
                
                for i in range(k):
                    if valid_add_mask(parts_mask[i], v):
                        if parts_mask[i] == 0:
                            if seen_empty:
                                continue
                            seen_empty = True
                        
                        parts_mask[i] |= v_bit
                        parts_list[i].append(v)
                        result = bt(idx + 1)
                        if result is not None:
                            return result
                        parts_mask[i] ^= v_bit
                        parts_list[i].pop()
                return None
            
            result = bt(0)
            if result is not None:
                return result
    
    # Fallback: greedy construction
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            part_mask = 0
            for u in part:
                part_mask |= (1 << u)
            if valid_add_mask(part_mask, v):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
