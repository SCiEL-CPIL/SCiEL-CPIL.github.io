import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Partition vertices
    A = list(range(10))
    B = list(range(10, 20))
    
    # Initialize adjacency matrix with -1
    adj = [[-1] * n for _ in range(n)]
    edges = []
    
    # Phase 1: Construct complete bipartite graph K_10,10
    # Coloring: c(a_i, b_j) = (i + j) % 20
    # This ensures every 4-cycle in the bipartite graph is rainbow.
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            c = (i + j) % num_colors
            adj[a][b] = c
            adj[b][a] = c
            edges.append([a, b, c])
    
    # List of remaining edges to try adding
    # These are edges within A and within B
    remaining_edges = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            remaining_edges.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            remaining_edges.append((B[i], B[j]))
    
    random.shuffle(remaining_edges)
    
    # Helper to check if adding edge (u, v) with color c is valid
    def can_add_edge(u, v, c):
        # Check all 4-cycles that would be formed by adding (u, v)
        # A 4-cycle involving (u, v) must be of the form u - a - b - v - u
        # where a is neighbor of u, b is neighbor of v, and (a, b) is an edge.
        # The colors must be distinct: c, adj[u][a], adj[a][b], adj[b][v]
        
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[u][a] == -1:
                continue
            ca = adj[u][a]
            
            # If ca == c, we have a conflict immediately in any cycle u-a-...-v-u
            # Wait, the cycle is u-a-b-v-u.
            # Edges: (u,a), (a,b), (b,v), (v,u).
            # Colors: ca, adj[a][b], adj[b][v], c.
            # We need {ca, adj[a][b], adj[b][v], c} to be distinct.
            
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj[v][b] == -1:
                    continue
                if adj[a][b] == -1:
                    continue
                
                cb = adj[a][b]
                cv = adj[v][b]
                
                # Check distinctness
                if ca == c or cb == c or cv == c or ca == cb or ca == cv or cb == cv:
                    return False
        
        return True

    # Phase 2: Greedily add intra-partition edges
    for u, v in remaining_edges:
        if time.time() > deadline:
            break
        
        # Try each color
        # Heuristic: try colors that are less frequent or random
        colors = list(range(num_colors))
        random.shuffle(colors)
        
        added = False
        for c in colors:
            if can_add_edge(u, v, c):
                adj[u][v] = c
                adj[v][u] = c
                edges.append([u, v, c])
                added = True
                break
        
        # If no color worked, skip this edge
        
    # Phase 3: Local Search / Recoloring to unlock more edges
    # Try to recolor existing edges to allow more additions
    # This is expensive, so we only do a few iterations
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 10:
        improved = False
        iterations += 1
        
        # Try to add any remaining edge that was previously skipped
        # Re-evaluate remaining edges
        current_edge_set = set((min(u, v), max(u, v)) for u, v, c in edges)
        all_possible = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in current_edge_set:
                    all_possible.append((i, j))
        
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline - 5:
                break
            
            # Try to add (u, v)
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for c in colors:
                if can_add_edge(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    improved = True
                    added = True
                    break
            
            if added:
                # If we added an edge, we might need to restart the loop to find more
                break
    
    # Final Verification
    # Check if the result is valid
    def check_validity():
        for u in range(n):
            for v in range(n):
                if u == v or adj[u][v] == -1:
                    continue
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    if not check_validity():
        # Fallback to just the bipartite part if something went wrong
        # (Should not happen if can_add_edge is correct)
        # Reconstruct bipartite only
        adj = [[-1] * n for _ in range(n)]
        edges = []
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                c = (i + j) % num_colors
                adj[a][b] = c
                adj[b][a] = c
                edges.append([a, b, c])

    return edges
