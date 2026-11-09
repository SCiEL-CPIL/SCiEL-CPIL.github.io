import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_edges_count = 0
    best_graph = None
    
    n = 16
    # List of all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    # Bitmask helper for 3-colorability
    def is_3_colorable(adj):
        # adj is a list of 16 ints (bitmasks)
        # Use backtracking with bitmasks
        color = [-1] * n
        # Order vertices by degree descending to prune faster
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Check available colors
            forbidden = 0
            # Iterate through bits set in adj[v]
            mask = adj[v]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    forbidden |= (1 << color[u])
                mask ^= lsb
                
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
            color[v] = -1
            return False
            
        return backtrack(0)

    # Main search loop
    # We can also try to bias towards denser graphs?
    # Random maximal triangle-free graphs are usually quite dense.
    
    while time.time() < deadline:
        # Randomize edge order
        random.shuffle(all_edges)
        
        # Build maximal triangle-free graph
        adj = [0] * n
        edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a triangle
            if (adj[u] & adj[v]) == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                
        # Check if this graph is 3-colorable
        if not is_3_colorable(adj):
            if len(edges) > best_edges_count:
                best_edges_count = len(edges)
                best_graph = edges.copy()
                
    # If no 4-chromatic graph found (unlikely), return Grötzsch as fallback
    if best_graph is None:
        # Grötzsch graph
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    return best_graph
