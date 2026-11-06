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
                
                # Filter for C4-safe vertices
                safe_vertices = []
                for v in non_neigh:
                    if (adj2[u] & adj[v]) == 0:
                        safe_vertices.append(v)
                
                if not safe_vertices:
                    # Could not add any edge to u.
                    # Check if any other vertex can add an edge?
                    # For efficiency, we break and check min degree.
                    # If we want to be thorough, we could iterate all vertices,
                    # but that's expensive.
                    break
                
                # Strategy: Pick v from safe_vertices with lowest degree
                # This helps balance the graph and raise min degree.
                # To maintain some randomness, pick randomly from the k lowest degree vertices.
                k = min(5, len(safe_vertices))
                # Sort by degree
                safe_vertices.sort(key=lambda x: degrees[x])
                # Pick randomly from the first k
                v = random.choice(safe_vertices[:k])
                
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                
                # Update adj2
                adj2[u] |= adj[v]
                adj2[v] |= adj[u]
                
                # For all w in N(u), adj2[w] gains v
                # For all x in N(v), adj2[x] gains u
                
                temp_u = adj[u] & ~(1 << v) # exclude v
                while temp_u:
                    lsb = temp_u & -temp_u
                    w = lsb.bit_length() - 1
                    adj2[w] |= (1 << v)
                    temp_u ^= lsb
                            
                temp_v = adj[v] & ~(1 << u) # exclude u
                while temp_v:
                    lsb = temp_v & -temp_v
                    x = lsb.bit_length() - 1
                    adj2[x] |= (1 << u)
                    temp_v ^= lsb
                            
                added = True
                # Loop continues to pick next min-degree vertex
            
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
