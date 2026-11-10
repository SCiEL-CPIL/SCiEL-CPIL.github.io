import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return x.bit_count()

    def greedy_edge_packing(adj, n, deadline):
        # Collect all edges
        edges = []
        for u in range(n):
            mask = adj[u] & ~((1 << u) - 1) # Only v > u
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                edges.append((u, v))
                mask ^= lsb
        
        # Heuristic sorting: 
        # We want to process edges that are "harder" to fit first? 
        # Or edges that are in dense areas first?
        # In First-Fit, the order matters.
        # A common heuristic for edge coloring/packing is to sort by the sum of degrees of endpoints in descending order.
        # This tends to put high-degree vertices' edges early, potentially filling cliques that involve high-degree vertices.
        
        # Let's try sorting by (deg[u] + deg[v]) descending.
        degs = [count_bits(adj[i]) for i in range(n)]
        edges.sort(key=lambda e: degs[e[0]] + degs[e[1]], reverse=True)
        
        # Also try a few different orderings to see if we can find a better packing?
        # Given the 120s limit, we can afford to run a few passes with different heuristics and keep the best.
        
        best_cliques = None
        best_count = float('inf')
        
        # Strategy 1: Sort by degree sum descending
        # Strategy 2: Sort by degree sum ascending
        # Strategy 3: Random shuffle (multiple times)
        
        strategies = [
            ("deg_desc", edges),
            ("deg_asc", sorted(edges, key=lambda e: degs[e[0]] + degs[e[1]])),
        ]
        
        # Add a few random shuffles if time permits
        # But let's first just implement the core packing for these.
        
        for name, edge_list in strategies:
            if time.time() > deadline:
                break
                
            # Data structures for packing
            # cliques: list of (vertex_mask, vertex_list)
            # We need to check if an edge (u,v) can be added to a clique C.
            # Condition: u and v must be connected to ALL vertices in C, and u-v is an edge (true).
            # Also, u and v must be connected to each other (true).
            # So, for clique with mask M:
            #   (adj[u] & M) == M  AND  (adj[v] & M) == M
            #   Also, u and v must not already be in the clique in a way that breaks it? 
            #   Actually, if u is in C, then v must be connected to all in C. 
            #   If neither u nor v is in C, they must both be connected to all in C.
            
            cliques = [] # List of (mask, list)
            
            # To speed up search, we can iterate through existing cliques.
            # Since n is small (32, 64), the number of cliques won't be huge (maybe < 100).
            # Checking each clique is O(1) with bitmasks.
            
            for u, v in edge_list:
                if time.time() > deadline:
                    break
                
                placed = False
                # Try to fit into existing cliques
                # Optimization: Check cliques that contain u or v first?
                # Or just check all?
                # Checking all is O(K) where K is num cliques.
                
                for i in range(len(cliques)):
                    mask, vlist = cliques[i]
                    # Check if u and v can be added
                    # If u is already in mask, we just need v to connect to all in mask.
                    # If v is already in mask, we just need u to connect to all in mask.
                    # If neither, both must connect to all in mask.
                    
                    # Note: It is possible that u is in mask but v is not, or vice versa.
                    # If u in mask:
                    #   We need (adj[v] & mask) == mask.
                    #   Also v must be connected to u (true).
                    #   But wait, if u is in mask, the clique already contains u. 
                    #   Adding v means the new clique is C U {v}.
                    #   v must be connected to all vertices in C.
                    
                    # If v in mask:
                    #   We need (adj[u] & mask) == mask.
                    
                    # If neither:
                    #   We need (adj[u] & mask) == mask AND (adj[v] & mask) == mask.
                    
                    u_in = (mask >> u) & 1
                    v_in = (mask >> v) & 1
                    
                    if u_in and v_in:
                        # Edge already covered? No, we process each edge once.
                        # But if both are in the clique, the edge is already part of the clique's internal structure?
                        # No, the clique is a set of vertices. The edges *within* the clique are covered.
                        # If u and v are both in the clique, the edge (u,v) is already "covered" by this clique?
                        # Yes, if u and v are in the clique, the edge (u,v) is part of the clique.
                        # But we are assigning edges to cliques. 
                        # If we assign edge (u,v) to a clique that already contains u and v, that's fine.
                        # However, in our construction, we build cliques by adding vertices.
                        # If a clique contains {u, v, ...}, then all edges between them are covered.
                        # So if u and v are both in mask, this clique already covers the edge (u,v).
                        # We should not assign the edge to a *new* clique if it's already covered.
                        # But our loop iterates over edges. 
                        # If we just append the edge to the clique, we don't change the clique's vertex set.
                        # But we must ensure every edge is in EXACTLY one clique.
                        # If u and v are in clique C, the edge (u,v) is in C.
                        # So we can mark it as placed.
                        placed = True
                        break
                    elif u_in:
                        # v must connect to all in mask
                        if (adj[v] & mask) == mask:
                            # Add v to clique
                            new_mask = mask | (1 << v)
                            new_vlist = vlist + [v]
                            cliques[i] = (new_mask, new_vlist)
                            placed = True
                            break
                    elif v_in:
                        # u must connect to all in mask
                        if (adj[u] & mask) == mask:
                            new_mask = mask | (1 << u)
                            new_vlist = vlist + [u]
                            cliques[i] = (new_mask, new_vlist)
                            placed = True
                            break
                    else:
                        # Both must connect to all in mask
                        if (adj[u] & mask) == mask and (adj[v] & mask) == mask:
                            new_mask = mask | (1 << u) | (1 << v)
                            new_vlist = vlist + [u, v]
                            cliques[i] = (new_mask, new_vlist)
                            placed = True
                            break
                
                if not placed:
                    # Create new clique
                    cliques.append((1 << u | 1 << v, [u, v]))
            
            # End of edge loop for this strategy
            if time.time() > deadline:
                continue
                
            # Count cliques
            # Note: Some cliques might have size < 2? No, we start with 2 and add.
            # But wait, the problem says "at least two vertices".
            # Our construction ensures size >= 2.
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                # Store only the vertex lists
                best_cliques = [c[1] for c in cliques]
                
        # If we didn't find any (e.g. timeout in first strategy), return empty or fallback
        if best_cliques is None:
             # Fallback: just return pairs?
             # This should not happen if deadline is generous.
             return []
             
        return best_cliques

    start_time = time.time()
    # Reserve 5s for final processing
    deadline = start_time + 115
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    cliques_1 = greedy_edge_packing(adj_1, 32, deadline)
    cliques_2 = greedy_edge_packing(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
