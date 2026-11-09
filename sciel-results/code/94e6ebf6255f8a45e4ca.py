import time
import random
import itertools

def construct():
    n = 18
    # Graph representation: set of edges, or adjacency matrix
    # Using adjacency matrix for speed
    adj = [[0] * n for _ in range(n)]
    
    def add_edge(u, v):
        if u == v: return
        if adj[u][v]: return
        # Check if adding (u,v) creates a K4
        # Common neighbors of u and v
        common = [w for w in range(n) if adj[u][w] and adj[v][w]]
        for w in common:
            for x in common:
                if w != x and adj[w][x]:
                    return False
        adj[u][v] = 1
        adj[v][u] = 1
        return True

    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0

    def is_k4_free():
        # Not needed if we maintain it, but good for debugging
        pass

    def get_score():
        # Find largest triangle-free induced subgraph
        # Bitmask representation
        best = 0
        # Iterate over all masks
        # 2^18 = 262144
        for mask in range(1 << n):
            # Check if mask is triangle-free
            # For each vertex in mask, check if it has two neighbors in mask that are connected
            # Or simply: no triangle exists in the induced subgraph
            # A triangle is a set of 3 vertices all connected.
            # We can check for any triangle in the mask.
            # If no triangle, it's triangle-free.
            
            # Optimization: if popcount(mask) <= best, skip
            if bin(mask).count('1') <= best:
                continue
                
            # Check for triangle
            has_triangle = False
            # Iterate over triples in mask
            # This is slow in Python.
            # Better: for each vertex i in mask, check neighbors j,k in mask.
            vertices = [i for i in range(n) if (mask >> i) & 1]
            for i in range(len(vertices)):
                u = vertices[i]
                # Neighbors of u in mask
                neigh_u = [v for v in vertices[i+1:] if adj[u][v]]
                for j in range(len(neigh_u)):
                    v = neigh_u[j]
                    for w in neigh_u[j+1:]:
                        if adj[v][w]:
                            has_triangle = True
                            break
                    if has_triangle:
                        break
                if has_triangle:
                    break
            
            if not has_triangle:
                best = bin(mask).count('1')
                
        return best

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # 1. Initialize
    # Start with a random graph and remove K4s
    # Or start empty and add random edges
    random.seed(42)
    
    # Try to build a dense K4-free graph
    # Start with all possible edges, then remove those that form K4?
    # No, that's hard.
    # Let's start with a random graph G(18, 0.5) and remove edges to make it K4-free.
    
    # Initialize adj with random edges
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                adj[u][v] = 1
                adj[v][u] = 1
                
    # Remove K4s
    # Iterate until no K4s
    changed = True
    while changed:
        changed = False
        # Find all K4s
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if not adj[a][d] or not adj[b][d] or not adj[c][d]: continue
                        # Found K4 {a,b,c,d}
                        # Remove a random edge from it
                        # e.g. remove (c,d)
                        remove_edge(c, d)
                        changed = True
                        break
                    if changed: break
                if changed: break
            if changed: break

    # 2. Local Search
    start_time = time.time()
    timeout = 100.0 # Keep safe margin
    
    current_score = get_score()
    
    while time.time() - start_time < timeout:
        # Find largest triangle-free set
        # We need the set itself, not just the size, to try adding edges
        best_mask = 0
        best_size = 0
        for mask in range(1 << n):
            if bin(mask).count('1') <= best_size:
                continue
            # Check triangle-free
            vertices = [i for i in range(n) if (mask >> i) & 1]
            has_triangle = False
            for i in range(len(vertices)):
                u = vertices[i]
                neigh_u = [v for v in vertices[i+1:] if adj[u][v]]
                for j in range(len(neigh_u)):
                    v = neigh_u[j]
                    for w in neigh_u[j+1:]:
                        if adj[v][w]:
                            has_triangle = True
                            break
                    if has_triangle:
                        break
                if has_triangle:
                    break
            
            if not has_triangle:
                best_size = bin(mask).count('1')
                best_mask = mask
        
        # Try to add edges within best_mask
        vertices_in_set = [i for i in range(n) if (best_mask >> i) & 1]
        improved = False
        for i in range(len(vertices_in_set)):
            for j in range(i+1, len(vertices_in_set)):
                u = vertices_in_set[i]
                v = vertices_in_set[j]
                if adj[u][v]:
                    continue
                # Check if adding (u,v) creates K4
                # Common neighbors
                common = [w for w in range(n) if adj[u][w] and adj[v][w]]
                safe = True
                for w in common:
                    for x in common:
                        if w != x and adj[w][x]:
                            safe = False
                            break
                    if not safe:
                        break
                if safe:
                    # Add edge
                    add_edge(u, v)
                    new_score = get_score()
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove_edge(u, v)
            
            if time.time() - start_time > timeout:
                break
        if not improved:
            # Try random restart or perturb?
            # Or break if no improvement
            break
            
    return get_edges()
