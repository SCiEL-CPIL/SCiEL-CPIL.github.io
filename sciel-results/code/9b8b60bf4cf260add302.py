import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    edges = []
    
    # Construct Clebsch Graph
    # Vertices are 4-bit integers 0..15
    # Edge exists if Hamming distance is 1 or 4
    
    def hamming_distance(a, b):
        return bin(a ^ b).count('1')
        
    for u in range(n):
        for v in range(u + 1, n):
            if hamming_distance(u, v) in (1, 4):
                edges.append([u, v])
                
    # Verification: Triangle Free
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    is_triangle_free = True
    for u in range(n):
        for v in adj[u]:
            if v < u: continue # Check each edge once? No, check neighbors of u
            # For each pair of neighbors of u, check if they are connected
            # Actually, simpler: for each edge (u,v), check if any common neighbor
            pass
            
    # More efficient triangle check:
    # For each vertex u, check if any two neighbors are connected
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    is_triangle_free = False
                    break
            if not is_triangle_free:
                break
        if not is_triangle_free:
            break
    
    if not is_triangle_free:
        # Fallback to a known valid graph (Grötzsch) if Clebsch logic fails
        # But Clebsch logic is deterministic and correct.
        # This is just a safety net.
        return [] 

    # Verification: Not 3-Colorable
    # Use backtracking to try to 3-color the graph
    colors = [-1] * n
    
    def can_color(idx):
        if idx == n:
            return True
        for c in range(3):
            if all(colors[neighbor] != c for neighbor in adj[idx]):
                colors[idx] = c
                if can_color(idx + 1):
                    return True
                colors[idx] = -1
        return False

    # Optimize coloring check with ordering (highest degree first)
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Map original indices to positions in order for coloring?
    # Simpler: just use the standard backtracking, n=16 is small.
    
    # To speed up, we can use bitmasks or just simple recursion.
    # 16 vertices, degree ~5. 3^16 is too big for brute force, but backtracking with pruning is fast.
    
    def is_3_colorable():
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        verts = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        def backtrack(i):
            if i == n:
                return True
            v = verts[i]
            for c in range(3):
                # Check if color c is available for v
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(i + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    if is_3_colorable():
        # If it is 3-colorable, it's invalid for the task (needs chi >= 4)
        # The Clebsch graph is NOT 3-colorable.
        # If this returns True, there is a bug in the graph construction or coloring logic.
        return []

    return edges
