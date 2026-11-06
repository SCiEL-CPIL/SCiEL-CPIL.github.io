import time
import random
import numpy as np
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Strategy: 
    # 1. Initialize with a dense random graph (p=0.5).
    # 2. Repair phase: Iteratively remove edges that are part of C6s until C6-free.
    # 3. Improve phase: Try to add edges back if they don't create C6s.
    # 4. Use bitsets for fast C6 detection/removal? 
    #    Actually, for n=40, checking if a specific edge is in a C6 can be done by checking 
    #    if there is a path of length 4 between its endpoints in the graph WITHOUT that edge?
    #    No, simpler: 
    #    A graph is C6-free. To check if adding (u,v) creates a C6, check if dist(u,v) == 4 in current graph?
    #    No, check if there is a path of length 4. 
    #    Using A^4 is fast.
    
    # Let's try a different initialization: 
    # Start with a random graph with high density, then repair.
    # Repair: Find a C6, remove a random edge from it. Repeat until no C6s.
    # This might yield a different local optimum than greedy addition.
    
    # Even better: Use the "Spectral" idea from the prompt? 
    # "Use a spectral initialization... edges are added with probability proportional to the product of vertex spectral components"
    # This is vague. Let's stick to a robust "Repair then Refine" strategy.
    
    # Step 1: Generate a random graph G with p=0.6
    # Step 2: While G has a C6:
    #    Find a C6 (using BFS or matrix powers to identify one).
    #    Remove a random edge from that C6.
    # Step 3: Refine:
    #    Try to add random non-edges if A^4[u][v] == 0.
    #    Try to swap edges to increase density? (Maybe not needed if we start dense)
    
    # Finding a C6 efficiently:
    # We can use the fact that if A^4 has an entry > 0 for u!=v, there is a walk of length 4.
    # We need a *simple* path of length 4 to form a C6 with edge (u,v) if (u,v) exists.
    # Or, if we are repairing, we just need to find ANY C6.
    # A C6 exists if there are u,v with edge (u,v) and a path of length 4 between them.
    # We can iterate over all edges (u,v) and check if A^4[u][v] > 0 in the graph *without* (u,v)?
    # No, if (u,v) is in the graph, and there is a path of length 4 between u and v in the graph *including* (u,v), 
    # that path might use (u,v). 
    # Correct check: Remove (u,v), check if dist(u,v) == 4? 
    # Or simpler: Check if (A^4)[u][v] > 0 in the graph where (u,v) is removed?
    # That's O(E * n^3) which is too slow.
    
    # Alternative C6 detection:
    # Use BFS from each vertex to find cycles? 
    # For n=40, we can do a full C6 check by iterating over all pairs (u,v) and checking if there is a path of length 4.
    # But we need to ensure the path doesn't use (u,v).
    
    # Let's use the matrix method for the *repair* step:
    # 1. Compute A.
    # 2. Compute A^4.
    # 3. For each edge (u,v), if A^4[u][v] > 0, it's a candidate. 
    #    But A^4 includes walks that might use (u,v).
    #    If a walk of length 4 uses (u,v), it's not a simple path of length 4 avoiding (u,v).
    #    However, if A^4[u][v] > 0, it's likely a C6 exists.
    #    To be safe, we can just pick a random edge (u,v) where A^4[u][v] > 0 and remove it.
    #    This is a "blind" repair. It will eventually remove enough edges to break all C6s.
    
    # Let's implement this "Blind Repair" strategy.
    
    def has_c6(adj_np):
        # Returns True if there is likely a C6
        A2 = adj_np @ adj_np
        A4 = A2 @ A2
        # Check if any off-diagonal entry in A4 is > 0 for an existing edge?
        # Actually, if A4[u][v] > 0 and adj_np[u][v] == 1, it suggests a C6.
        # We need to be careful about walks that use the edge (u,v).
        # But for a quick heuristic repair, this is good enough.
        mask = (adj_np == 1)
        np.fill_diagonal(mask, 0)
        candidates = np.where((A4 > 0) & mask)
        return len(candidates[0]) > 0, candidates

    def remove_random_c6_edge(adj, adj_np, candidates):
        # Pick a random candidate edge (u,v) and remove it
        idx = random.randint(0, len(candidates[0]) - 1)
        u = candidates[0][idx]
        v = candidates[1][idx]
        adj[u][v] = 0
        adj[v][u] = 0
        adj_np[u][v] = 0
        adj_np[v][u] = 0

    # Main Loop
    best_edges = []
    best_count = 0
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # 1. Initialize random graph
        p_init = 0.5 + 0.1 * restart # Increase density with restarts?
        if p_init > 0.7: p_init = 0.7
        
        adj = [[0]*n for _ in range(n)]
        adj_np = np.zeros((n,n), dtype=np.int8)
        edges = []
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p_init:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    adj_np[u][v] = 1
                    adj_np[v][u] = 1
                    edges.append([u,v])
        
        # 2. Repair Phase
        # Remove edges until C6-free
        repair_steps = 0
        while time.time() < deadline - 5:
            has, candidates = has_c6(adj_np)
            if not has:
                break
            remove_random_c6_edge(adj, adj_np, candidates)
            # Remove from edges list
            # This is slow O(E). Let's rebuild edges at the end.
            repair_steps += 1
            if repair_steps > 1000:
                break
        
        # Rebuild edges list
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u,v])
        
        current_count = len(edges)
        
        # 3. Refine Phase: Try to add edges
        # Compute A4
        A2 = adj_np @ adj_np
        A4 = A2 @ A2
        
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
        
        no_improve = 0
        for step in range(2000):
            if time.time() > deadline - 2:
                break
                
            # Try to add a random non-edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) in edge_set:
                continue
                
            if A4[u][v] == 0:
                # Safe to add
                adj[u][v] = 1
                adj[v][u] = 1
                adj_np[u][v] = 1
                adj_np[v][u] = 1
                edges.append([u,v])
                edge_set.add((u,v))
                edge_set.add((v,u))
                current_count += 1
                no_improve = 0
                
                # Update A4
                A2 = adj_np @ adj_np
                A4 = A2 @ A2
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(edges)
                    
                if best_count >= 105:
                    return best_edges
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        # End of restart

    return best_edges
