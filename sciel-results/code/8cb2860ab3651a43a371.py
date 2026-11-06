import random
import time

def construct():
    n = 26
    num_edges = n * (n - 1) // 2
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Map edge index to (u, v)
    # colors[i] is the color of edges[i]
    # Initialize randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency lists for each color
    # adj[c][u] = set of neighbors of u in color c
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u].add(v)
        adj[c][v].add(u)
        
    # Function to calculate number of C4s containing edge (u, v) in color c
    # This is the number of paths u-a-b-v in G_c (excluding u-v)
    # i.e., count pairs (a, b) such that a in adj[c][u], b in adj[c][v], and a in adj[c][b]
    def count_c4s_containing_edge(c, u, v):
        count = 0
        # Iterate over neighbors of u
        for a in adj[c][u]:
            if a == v:
                continue
            # Check if a is connected to any neighbor of v
            # Neighbors of v
            for b in adj[c][v]:
                if b == u:
                    continue
                if b in adj[c][a]:
                    count += 1
        return count

    # Calculate total number of C4s
    # Note: Each C4 contains 4 edges. So summing count_c4s_containing_edge over all edges counts each C4 exactly 4 times.
    # We just want to minimize this sum.
    def get_total_c4_weight():
        total = 0
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            total += count_c4s_containing_edge(c, u, v)
        return total

    # Initial total
    current_total = get_total_c4_weight()
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Local Search
    while time.time() - start_time < timeout:
        # Pick a random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        c_old = colors[idx]
        
        # Pick a random new color
        c_new = random.randint(0, 4)
        while c_new == c_old:
            c_new = random.randint(0, 4)
            
        # Calculate delta
        # Removing (u,v) from c_old reduces C4s by count_c4s_containing_edge(c_old, u, v)
        # Adding (u,v) to c_new increases C4s by count_c4s_containing_edge(c_new, u, v)
        # Note: The count for c_new is calculated BEFORE adding the edge.
        # Is this correct?
        # Yes, any C4 formed by adding (u,v) must use (u,v) and a path u-a-b-v in the existing graph.
        
        delta = count_c4s_containing_edge(c_new, u, v) - count_c4s_containing_edge(c_old, u, v)
        
        if delta < 0 or (delta == 0 and random.random() < 0.5):
            # Accept move
            colors[idx] = c_new
            
            # Update adjacency lists
            adj[c_old][u].remove(v)
            adj[c_old][v].remove(u)
            adj[c_new][u].add(v)
            adj[c_new][v].add(u)
            
            current_total += delta
            
            if current_total == 0:
                # Valid coloring found
                break
                
    return {'n': n, 'colors': colors}
