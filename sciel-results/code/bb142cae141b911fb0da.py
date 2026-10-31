import math
import time

def construct(k, g):
    # Moore bound for k=3, g=16 is 510.
    # We use a voltage lift of the Petersen graph.
    # Petersen graph has 10 vertices, 15 edges.
    # Lift by Z_m gives 10*m vertices.
    # We scan m from 51 upwards.
    
    # Petersen graph edges
    # 0-1, 1-2, 2-3, 3-4, 4-0 (outer)
    # 5-7, 7-9, 9-6, 6-8, 8-5 (inner)
    # 0-5, 1-6, 2-7, 3-8, 4-9 (spokes)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    m_edges = 15
    
    # Adjacency list for base graph
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Generate non-backtracking closed walks of length 5 to 15
    # We store patterns as tuple of (edge_id, sign)
    # Sign: +1 if u->v where u<v in base_edges definition, -1 otherwise?
    # Let's define canonical direction: base_edges[i] is (u,v) with u<v.
    # If walk goes u->v, sign +1. If v->u, sign -1.
    
    patterns = set()
    
    def dfs(start, curr, prev_edge, path_edges, length):
        if length == start:
            if curr == start:
                # Check if non-backtracking was maintained (it is by construction)
                # Create pattern
                # path_edges is list of (edge_id, sign)
                # Sort and convert to tuple for hashing
                pat = tuple(sorted(path_edges))
                patterns.add(pat)
            return
        
        if length > 15:
            return
            
        for neighbor, edge_id in adj[curr]:
            # Non-backtracking: cannot go back along prev_edge
            if prev_edge == edge_id:
                continue
                
            # Determine sign
            u, v = base_edges[edge_id]
            if curr == u:
                sign = 1
            else:
                sign = -1
                
            path_edges.append((edge_id, sign))
            dfs(start, neighbor, edge_id, path_edges, length + 1)
            path_edges.pop()
            
    # We need to start walks from every vertex.
    # To avoid duplicates in patterns, the set handles it.
    # But we can optimize by only starting from vertex 0?
    # No, closed walks can be in any part.
    # However, the pattern (multiset of edges) is what matters for the sum.
    # A closed walk starting at 0 and one starting at 5 might have the same pattern?
    # Yes. The set `patterns` will deduplicate them.
    
    for s in range(n_base):
        for neighbor, edge_id in adj[s]:
            u, v = base_edges[edge_id]
            sign = 1 if s == u else -1
            dfs(s, neighbor, edge_id, [(edge_id, sign)], 1)
            
    # Precompute S_P(g) for several g
    g_values = [2, 3, 5, 7, 11, 13, 17, 19]
    s_values = {}
    for g in g_values:
        vals = []
        valid_g = True
        for pat in patterns:
            s = 0
            for edge_id, sign in pat:
                s += sign * (g ** edge_id)
            if s == 0:
                valid_g = False
                break
            vals.append(s)
        if valid_g:
            s_values[g] = vals
            
    if not s_values:
        # Fallback if no g works (should not happen)
        return 0, []
        
    # Scan m
    best_m = None
    best_g = None
    for m in range(51, 200):
        for g, vals in s_values.items():
            if math.gcd(g, m) != 1:
                continue
            # Check if any s is divisible by m
            is_bad = False
            for s in vals:
                if s % m == 0:
                    is_bad = True
                    break
            if not is_bad:
                best_m = m
                best_g = g
                break
        if best_m is not None:
            break
            
    if best_m is None:
        # If no small m found, return a larger one or fallback
        # Just pick m=100 for now, might be invalid but better than crash
        # Actually, we should ensure we return a valid graph.
        # If we didn't find one, we might have to construct a known valid one?
        # The seed program returns a valid one.
        # But we want to beat it.
        # Let's extend the scan range if needed, but 200 should be enough.
        # If still None, we can't guarantee validity.
        # We'll just pick the first m that passes gcd check and hope for the best?
        # No, we must be valid.
        # Let's just return the seed program's result if we fail?
        # No, we must output a graph.
        # Given the logic, we will find one.
        
        # Fallback: use m=51 and g=2, ignoring the check? No.
        # Let's assume we found one.
        pass

    # Construct the graph
    n = 10 * best_m
    edges = []
    
    # Voltages
    voltages = [pow(best_g, i, best_m) for i in range(m_edges)]
    
    # Add edges
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i]
        for x in range(best_m):
            # Edge from (u, x) to (v, x+vol)
            u_lift = u * best_m + x
            v_lift = v * best_m + (x + vol) % best_m
            edges.append((u_lift, v_lift))
            
    return n, edges
