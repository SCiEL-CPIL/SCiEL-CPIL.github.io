import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate Paley coloring for n=17
    n_paley = 17
    squares = {x * x % 17 for x in range(1, 17)}
    # colors[i, j] for i < j
    # We use a list of lists for adjacency or a flat list.
    # Flat list is required for output.
    # Let's use a 2D list for internal logic, then flatten.
    
    # Initialize base graph
    n_current = n_paley
    # adj[i][j] = 0 or 1
    adj = [[0] * n_current for _ in range(n_current)]
    for i in range(n_current):
        for j in range(i + 1, n_current):
            if (j - i) % 17 in squares:
                adj[i][j] = 0
            else:
                adj[i][j] = 1
            adj[j][i] = adj[i][j]
            
    best_n = n_current
    best_adj = [row[:] for row in adj]
    
    # Helper to check if a new vertex k with given colors to 0..k-1 creates a K2,2,2
    # colors: list of length k, colors[i] is color of (k, i)
    def is_valid_extension(k, colors, adj):
        # Check all triples of disjoint pairs from 0..k-1
        # This is expensive if done naively.
        # But k is small (up to 28).
        # Number of vertices in base: k.
        # We need to find if there exist u, v, x, y, z, w in 0..k-1
        # such that {u,v}, {x,y}, {z,w} are disjoint pairs
        # and all cross edges are color c, and edges from k to all 6 are color c.
        
        # Optimization:
        # For each color c in {0, 1}:
        # Let S_c = { i in 0..k-1 | colors[i] == c }
        # For each pair (u, v) in S_c:
        #   Let N_c(u, v) = { i in S_c \ {u,v} | adj[u][i]==c and adj[v][i]==c }
        #   We need to find two disjoint pairs {x,y} and {z,w} in N_c(u,v)
        #   such that all 4 edges between {x,y} and {z,w} are c.
        #   This is equivalent to checking if the subgraph induced by N_c(u,v) in color c
        #   contains a K_{2,2}.
        
        for c in [0, 1]:
            S_c = [i for i in range(k) if colors[i] == c]
            # If |S_c| < 4, no K2,2,2 possible with k in part 1?
            # K2,2,2 needs 6 vertices. k is one. Need 5 others.
            # But k is in a pair {k, u}. So we need u in S_c.
            # And we need 4 others in S_c.
            # So |S_c| must be at least 5.
            if len(S_c) < 5:
                continue
                
            # Iterate over all pairs (u, v) in S_c?
            # No, k is in one part. Let's say part 1 is {k, u}.
            # Then u must be in S_c.
            # The other 4 vertices {x,y,z,w} must be in S_c \ {u}.
            # And they must be connected to u with color c.
            # And they must form two pairs {x,y}, {z,w} with all cross-edges c.
            
            for u in S_c:
                # Candidates for the other 4 vertices
                candidates = [i for i in S_c if i != u and adj[u][i] == c]
                if len(candidates) < 4:
                    continue
                
                # Check if candidates contain a K_{2,2}
                # A K_{2,2} is a 4-cycle.
                # We can check all pairs of pairs in candidates.
                # Number of candidates m <= 27.
                # Pairs in candidates: O(m^2).
                # For each pair (x,y), check if there is another pair (z,w) disjoint
                # such that adj[x][z]==c, adj[x][w]==c, adj[y][z]==c, adj[y][w]==c.
                
                # Optimization: Precompute for each x in candidates, the set of neighbors in candidates with color c.
                # neigh[x] = { z in candidates | adj[x][z] == c }
                
                neigh = {}
                for x in candidates:
                    neigh[x] = set(z for z in candidates if z != x and adj[x][z] == c)
                
                # Now check for K_{2,2}
                # Iterate over pairs (x, y) in candidates
                for i in range(len(candidates)):
                    x = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        y = candidates[j]
                        # Common neighbors of x and y in candidates
                        common = neigh[x] & neigh[y]
                        # We need two disjoint vertices z, w in common
                        # such that z != w and z, w not in {x,y} (already ensured by candidates def)
                        # And we need the edge (z,w) to be c?
                        # No, in K_{2,2}, the parts are {x,y} and {z,w}.
                        # Edges are x-z, x-w, y-z, y-w.
                        # We already checked x-z, x-w, y-z, y-w are c (via common neighbors).
                        # We do NOT require z-w to be c. In fact, in K_{2,2,2}, 
                        # z and w are in the same part, so no edge required.
                        # So if |common| >= 2, we have a K_{2,2}.
                        if len(common) >= 2:
                            return False
        return True

    # Backtracking extension
    def extend(k, adj):
        nonlocal best_n, best_adj
        if time.time() > deadline:
            return False
        
        if k == 29:
            best_n = 29
            best_adj = [row[:] for row in adj]
            return True
        
        # Try all 2^k assignments? No, use DFS with pruning.
        # We assign colors to (k, 0), (k, 1), ..., (k, k-1)
        # We do it recursively.
        
        # To speed up, we can try random orders or just standard.
        # Standard DFS might be slow.
        # Let's use a randomized approach: try 0 then 1, but with random restarts if stuck.
        
        # For simplicity and robustness, we'll use a limited backtracking.
        # If k is large, 2^k is big.
        # We'll use a heuristic: try to match the "density" or just random.
        
        # Actually, let's just do a simple DFS with a node limit per restart.
        
        colors = [0] * k
        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == -1:
                # All assigned
                if is_valid_extension(k, colors, adj):
                    # Commit
                    for i in range(k):
                        adj[k][i] = colors[i]
                        adj[i][k] = colors[i]
                    if extend(k + 1, adj):
                        return True
                    # Undo
                    for i in range(k):
                        adj[k][i] = 0
                        adj[i][k] = 0
                return False
            
            # Try 0
            colors[idx] = 0
            if dfs(idx - 1):
                return True
            # Try 1
            colors[idx] = 1
            if dfs(idx - 1):
                return True
            
            return False
        
        # Add node k to adj (initialize with 0)
        for i in range(k):
            adj[k][i] = 0
            adj[i][k] = 0
        
        if dfs(k - 1):
            return True
        
        # Undo
        for i in range(k):
            adj[k][i] = 0
            adj[i][k] = 0
        return False

    # We need to handle the adjacency matrix size dynamically
    # Initialize adj for max size 29
    adj = [[0] * 29 for _ in range(29)]
    # Copy base
    for i in range(n_paley):
        for j in range(n_paley):
            adj[i][j] = [[0] * n_paley for _ in range(n_paley)][i][j] # Re-use logic
            # Actually, let's just rebuild adj properly
    adj = [[0] * 29 for _ in range(29)]
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            if (j - i) % 17 in squares:
                adj[i][j] = 0
            else:
                adj[i][j] = 1
            adj[j][i] = adj[i][j]
            
    # Try to extend
    # We might need multiple restarts if the first path fails
    # But the graph is fixed, so the solution is unique or non-existent for a given order.
    # Since we don't know the order, we can try shuffling the "virtual" order?
    # No, the vertices are labeled 0..28.
    
    # If direct extension fails, we might not find 29.
    # But the problem states a record of 29 exists.
    
    if extend(n_paley, adj):
        pass
    else:
        # If we didn't reach 29, we return whatever we have (17)
        # Or we can try a different base?
        # For now, return best_n
        pass
        
    # Flatten best_adj to colors list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])
            
    return {'n': best_n, 'colors': colors}
