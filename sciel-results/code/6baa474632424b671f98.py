import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bitmask helpers
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # Precompute masks for vertices
    def mask_except(vertices):
        m = (1 << n) - 1
        for v in vertices:
            m &= ~(1 << v)
        return m

    def creates_q3(u, v):
        # Check if adding edge (u, v) creates a Q3
        # We assume (u, v) is NOT currently in adj
        # 1. Find all pairs (a, b) such that (u,a), (a,b), (b,v) are in adj
        #    This forms a square u-a-b-v-u
        #    a must be neighbor of u, b must be neighbor of v, and (a,b) must be edge.
        
        # Neighbors of u (excluding v, since (u,v) not in adj yet)
        neighbors_u = adj[u]
        # Neighbors of v
        neighbors_v = adj[v]
        
        # List of (a, b) pairs forming a square with (u,v)
        squares = []
        # Iterate a in neighbors_u
        # We can iterate over bits in neighbors_u
        a_mask = neighbors_u
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            # b must be in neighbors_v AND neighbors of a
            # b_mask = neighbors_v & adj[a]
            b_mask = neighbors_v & adj[a]
            while b_mask:
                b = (b_mask & -b_mask).bit_length() - 1
                b_mask &= b_mask - 1
                if a == b: continue
                squares.append((a, b))
        
        if len(squares) < 2:
            return False
            
        # We need to find two distinct squares (a,b) and (c,d) such that:
        # 1. {a,b} != {c,d} and {u,v,a,b} != {u,v,c,d} (vertices distinct)
        # 2. There exist e, f such that:
        #    - e connected to a and c
        #    - f connected to b and d
        #    - e connected to f
        #    - e, f distinct from {u,v,a,b,c,d}
        #    - e != f
        
        num_squares = len(squares)
        for i in range(num_squares):
            a, b = squares[i]
            for j in range(i + 1, num_squares):
                c, d = squares[j]
                
                # Check distinctness of vertices in the two squares
                # Vertices involved: u, v, a, b, c, d
                if a == c or a == d or b == c or b == d:
                    continue
                if a == b or c == d: # Already checked a!=b, c!=d in generation? 
                    # a != b because a in N(u), b in N(v), u!=v. 
                    # But a could be v? No, (u,v) not in adj.
                    # b could be u? No.
                    # So a,b,c,d are distinct from u,v.
                    # We just need a,b distinct from c,d.
                    continue
                
                # The 6 vertices used so far
                used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                
                # Find candidates for e: neighbors of a AND neighbors of c, excluding used
                e_candidates = adj[a] & adj[c] & ~used_mask
                
                if not e_candidates:
                    continue
                    
                # Find candidates for f: neighbors of b AND neighbors of d, excluding used
                f_candidates = adj[b] & adj[d] & ~used_mask
                
                if not f_candidates:
                    continue
                
                # We need e in e_candidates, f in f_candidates, e != f, and (e,f) in adj
                # Check if any e in e_candidates is connected to any f in f_candidates
                # We can iterate over e_candidates
                e_mask = e_candidates
                found = False
                while e_mask and not found:
                    e = (e_mask & -e_mask).bit_length() - 1
                    e_mask &= e_mask - 1
                    # Check if any neighbor of e is in f_candidates
                    # And e != f (guaranteed if we check intersection with f_candidates correctly? 
                    # No, e could be in f_candidates. If e is in f_candidates, 
                    # we need f != e. 
                    # So we check adj[e] & f_candidates. 
                    # If e is in f_candidates, adj[e] has bit e? No, simple graph.
                    # So adj[e] & f_candidates will not include e.
                    # So if there is any bit set in adj[e] & f_candidates, we have a valid pair.
                    if adj[e] & f_candidates:
                        found = True
                
                if found:
                    return True
                    
        return False

    best_edges = 0
    best_graph = set()
    
    # Local Search
    for restart in range(2000): # Adjust based on speed
        if time.time() > deadline:
            break
            
        # Initialize graph
        # Option 1: Random sparse graph
        # Option 2: Greedy construction
        # Greedy is usually better for initialization
        random.shuffle(all_edges)
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            if not creates_q3(u, v):
                current_graph.add(edge)
                # Update adj
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Local Improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            # To avoid O(E) scan every time, we can just iterate all edges.
            # With n=16, E=120. This is fast.
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                if not creates_q3(u, v):
                    current_graph.add(edge)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    break # Restart scan
            
            # 2. If no edge added, try a 1-remove 1-add move (random walk)
            if not improved:
                # Pick a random edge to remove
                if current_graph:
                    remove_edge = random.choice(list(current_graph))
                    u, v = remove_edge
                    current_graph.remove(remove_edge)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Try to add a different edge
                    # We can try to add the same edge back (no change) or a new one.
                    # Better: try to add any valid edge.
                    # Since we just removed one, the graph is less dense, so likely some edge can be added.
                    # We don't want to immediately add the same edge back unless it's the only option, 
                    # but adding it back is a valid move (no-op). 
                    # To ensure progress, we could try to add a *different* edge.
                    
                    added = False
                    # Shuffle edges to pick a random candidate
                    random.shuffle(all_edges)
                    for edge in all_edges:
                        if edge == remove_edge:
                            continue
                        if edge in current_graph:
                            continue
                        u2, v2 = edge
                        if not creates_q3(u2, v2):
                            current_graph.add(edge)
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            added = True
                            break
                    
                    # If we couldn't add a different edge, we might be stuck in a local minimum.
                    # We can try to add the removed edge back to restore state, 
                    # or just accept the state (which is the same as before minus the edge, 
                    # so we lost an edge). 
                    # If we lost an edge and couldn't add another, we are worse off.
                    # So, if not added, we should probably revert? 
                    # Or just continue? 
                    # If we revert, we loop forever on this state.
                    # Let's try to add the removed edge back if we can't find a better one.
                    if not added:
                        # Revert
                        current_graph.add(remove_edge)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        # To avoid infinite loop on this specific state, 
                        # we can break out of this restart if we can't improve?
                        # Or just try another random removal.
                        # For now, let's just break the local search for this restart.
                        break
                    else:
                        improved = True # We changed the graph (removed one, added another)
                        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
            
    return [[u, v] for u, v in best_graph]
