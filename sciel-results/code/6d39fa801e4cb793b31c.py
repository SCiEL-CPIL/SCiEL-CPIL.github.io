import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = 39
    best_edges = []
    
    def try_greedy(n, max_restarts=200):
        min_deg_req = n - 39
        if min_deg_req <= 0:
            return True, []
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return False, None
            
            # Bitsets for neighbors and neighbors-of-neighbors
            adj = [0] * n
            adj2 = [0] * n
            degrees = [0] * n
            
            # List of all potential edges
            # We will pick edges in a random order but prioritize low degree vertices?
            # Actually, the "pick min degree vertex" strategy is better.
            
            # To allow randomness, we can shuffle the vertex order initially?
            # Or just pick min degree, and if ties, random.
            
            # We will try to add edges until no more can be added.
            # Then check min degree.
            
            # To avoid getting stuck in a bad local optimum, we can randomize the 
            # order in which we check non-neighbors for a given vertex.
            
            while True:
                if time.time() > deadline - 5:
                    return False, None
                
                # Find vertex with minimum degree
                min_deg = min(degrees)
                candidates = [i for i in range(n) if degrees[i] == min_deg]
                u = random.choice(candidates)
                
                # Get non-neighbors of u
                # non_neigh_mask = ~adj[u] & ((1 << n) - 1)
                # We want to try connecting u to v if it doesn't create C4.
                
                # To add randomness, we can shuffle the non-neighbors.
                # But for n=46, iterating is fast.
                
                added = False
                # Create list of non-neighbors
                non_neigh = []
                mask = ((1 << n) - 1) ^ adj[u] ^ (1 << u) # exclude self and neighbors
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    non_neigh.append(v)
                    mask ^= lsb
                
                random.shuffle(non_neigh)
                
                for v in non_neigh:
                    # Check if adding (u,v) creates C4
                    # Condition: adj2[u] & adj[v] == 0
                    if (adj2[u] & adj[v]) == 0:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        
                        # Update adj2
                        # adj2[u] gains adj[v]
                        adj2[u] |= adj[v]
                        # adj2[v] gains adj[u]
                        adj2[v] |= adj[u]
                        
                        # For all w in N(u), adj2[w] gains v
                        # For all x in N(v), adj2[x] gains u
                        # Note: u is in N(v) now, so adj2[v] already updated?
                        # adj2[v] is neighbors of neighbors of v.
                        # Neighbors of v are N(v).
                        # So adj2[v] = union of N(z) for z in N(v).
                        # When we add (u,v), u becomes in N(v).
                        # So N(u) is added to adj2[v].
                        # This is done by adj2[v] |= adj[u].
                        
                        # Now for w in N(u) (old neighbors, excluding v which was not there):
                        # v is in N(u). So N(v) is added to adj2[w].
                        # Wait, adj2[w] = union of N(z) for z in N(w).
                        # If w is in N(u), then u is in N(w).
                        # Now v is in N(u). So v is a neighbor of a neighbor of w.
                        # So v should be in adj2[w].
                        # So for all w in N(u) (old), adj2[w] |= (1 << v).
                        
                        # Similarly, for all x in N(v) (old), adj2[x] |= (1 << u).
                        
                        # Iterate old N(u)
                        temp_u = adj[u] & ~(1 << v) # exclude v
                        while temp_u:
                            lsb = temp_u & -temp_u
                            w = lsb.bit_length() - 1
                            adj2[w] |= (1 << v)
                            temp_u ^= lsb
                            
                        # Iterate old N(v)
                        temp_v = adj[v] & ~(1 << u) # exclude u
                        while temp_v:
                            lsb = temp_v & -temp_v
                            x = lsb.bit_length() - 1
                            adj2[x] |= (1 << u)
                            temp_v ^= lsb
                            
                        added = True
                        break # Move to next min-degree vertex
                
                if not added:
                    # Could not add any edge to u.
                    # Try next min-degree vertex?
                    # If we iterate through all min-degree vertices and none can add, we are stuck.
                    # We can break and check min degree.
                    # To be thorough, we should check if ANY vertex can add an edge.
                    # But if u (min deg) can't, maybe a higher deg vertex can?
                    # Adding an edge to a higher deg vertex doesn't help min degree directly,
                    # but might enable future additions.
                    # However, if min degree vertex is stuck, it's likely the graph is saturated
                    # with respect to C4s for that vertex.
                    
                    # Let's check if any vertex can add an edge.
                    any_addable = False
                    for i in range(n):
                        mask = ((1 << n) - 1) ^ adj[i] ^ (1 << i)
                        # Check if any v in mask satisfies (adj2[i] & adj[v]) == 0
                        # This is expensive.
                        # Given the greedy nature, if the min-degree vertex is stuck,
                        # it's probable the graph is "full".
                        # We will break and evaluate.
                        pass
                    
                    break
            
            # Check min degree
            if min(degrees) >= min_deg_req:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return True, edges
                
        return False, None

    # Try from n=47 down to n=40
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
            
        success, edges = try_greedy(target_n, max_restarts=1000)
        if success:
            best_n = target_n
            best_edges = edges
            break
            
    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
