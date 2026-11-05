import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Grötzsch graph edges (0..10)
    # 5-cycle: 0-1-2-3-4-0
    # Mycielski:
    # New vertices 5..9 corresponding to 0..4
    # Vertex 10 connected to 5..9
    # Edges:
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Mycielski edges:
    # For each edge (u,v) in cycle, add (u, 5+v) and (v, 5+u)
    # And (5+u, 10) for u in 0..4
    
    base_edges = []
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in cycle:
        base_edges.append((u, v))
        base_edges.append((u, 5+v))
        base_edges.append((v, 5+u))
    for u in range(5):
        base_edges.append((5+u, 10))
        
    best_edges = []
    best_count = len(base_edges) # 20
    
    # Function to check 3-colorability
    def is_3_colorable(adj):
        # adj is list of 16 ints (bitmasks)
        # Backtracking
        n = 16
        # Order vertices by degree
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        
        # colors[i] = 0, 1, 2 or -1
        colors = [-1] * n
        
        # Map order index to vertex
        # We will assign colors in order of 'order'
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Check which colors are forbidden
            forbidden = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    forbidden |= (1 << colors[u])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def complete_graph(adj, edges_set):
        # Try to add all possible edges that don't create triangles
        # Iterate until no change
        changed = True
        while changed:
            changed = False
            for u in range(16):
                for v in range(u+1, 16):
                    if (u, v) in edges_set:
                        continue
                    # Check if adding (u,v) creates a triangle
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    if common == 0:
                        # Safe to add
                        edges_set.add((u, v))
                        edges_set.add((v, u)) # if using undirected set of tuples? 
                        # Let's use a set of frozensets or just (min, max)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
        return adj, edges_set

    def get_edges_from_adj(adj):
        edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Search loop
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        # Initialize adj
        adj = [0] * 16
        edges_set = set()
        
        # Add base edges
        for u, v in base_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges_set.add((min(u,v), max(u,v)))
            
        # Randomly connect extra vertices (11..15) to base (0..10) and each other
        # This introduces variety
        # We can randomly pick a subset of potential edges
        
        # Potential edges involving 11..15
        potential = []
        for u in range(11, 16):
            for v in range(16):
                if u != v:
                    potential.append((min(u,v), max(u,v)))
        
        # Shuffle potential
        random.shuffle(potential)
        
        # Add a random subset of potential edges if valid
        # To avoid getting stuck in sparse graphs, add many
        for u, v in potential:
            if (u, v) in edges_set:
                continue
            # Check triangle
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
                
        # Complete the graph
        adj, edges_set = complete_graph(adj, edges_set)
        
        # Check 3-colorability
        if not is_3_colorable(adj):
            current_edges = get_edges_from_adj(adj)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
                # Optional: update base? No, keep base fixed to ensure non-3-colorability seed?
                # Actually, if we start with Grötzsch, it's non-3-colorable.
                # Adding edges keeps it non-3-colorable.
                # So any completion of a graph containing Grötzsch is non-3-colorable.
                # So we don't even need to check 3-colorability if we always include Grötzsch!
                # Wait, does adding edges to a non-3-colorable graph always keep it non-3-colorable?
                # Yes. If G is not k-colorable, any supergraph H is not k-colorable.
                # So, if we start with Grötzsch, we are guaranteed non-3-colorable.
                # So we can skip the check! This saves time.
                
    return best_edges
