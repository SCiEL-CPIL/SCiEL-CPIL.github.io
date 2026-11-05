import time
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    best_edges = 0
    best_graph = []
    
    # Helper to check if adding edge (u,v) creates a triangle
    # adj is a list of sets or bitmasks
    # Using bitmasks for speed
    n = 16
    all_mask = (1 << n) - 1
    
    def is_triangle_free(adj, u, v):
        # Check if u and v have a common neighbor
        # adj[u] & adj[v] != 0
        return (adj[u] & adj[v]) == 0

    def add_edge_if_valid(adj, u, v):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False # Already exists
        if not is_triangle_free(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = 0
            while mask:
                if mask & 1:
                    if u < v:
                        edges.append([u, v])
                mask >>= 1
                v += 1
        return edges

    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def is_3_colorable(adj):
        # Backtracking search
        colors = [-1] * n
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def bt(idx):
            if idx == n:
                return True
            u = order[idx]
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if any neighbor has color c
                valid = True
                mask = adj[u]
                v = 0
                while mask:
                    if mask & 1:
                        if colors[v] == c:
                            valid = False
                            break
                    mask >>= 1
                    v += 1
                if valid:
                    colors[u] = c
                    if bt(idx + 1):
                        return True
                    colors[u] = -1
            return False
        
        return bt(0)

    # Main search loop
    while time.time() - start_time < time_limit:
        # 1. Generate random bipartite graph
        # Random partition
        partition = random.sample(range(n), n)
        part_a = set(partition[:8])
        part_b = set(partition[8:])
        
        adj = [0] * n
        # Add random edges between A and B
        # Density 0.9
        for u in part_a:
            for v in part_b:
                if random.random() < 0.9:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # 2. Mutation: Add an edge within a part to break bipartiteness
        # Pick a part with at least 2 vertices
        # Let's pick from part_a
        if len(part_a) >= 2:
            u, v = random.sample(list(part_a), 2)
            # Add edge (u, v)
            # This creates triangles with common neighbors in part_b
            # Common neighbors are all w in part_b connected to both u and v
            common_neighbors = adj[u] & adj[v] & (1 << 16) # Mask for part_b? 
            # Actually, just find common neighbors in the whole graph
            common = adj[u] & adj[v]
            
            # To maintain triangle-free, we must remove edges (u,w) or (v,w) for all w in common
            # Strategy: Remove (u,w) for all w in common? Or random?
            # Removing all ensures triangle-free.
            # Let's try removing all edges from u to common neighbors
            adj[u] &= ~common
            # Update neighbors
            for w in range(n):
                if common & (1 << w):
                    adj[w] &= ~(1 << u)
            
            # Now add the new edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            # If part_a is small, try part_b
            if len(part_b) >= 2:
                u, v = random.sample(list(part_b), 2)
                common = adj[u] & adj[v]
                adj[u] &= ~common
                for w in range(n):
                    if common & (1 << w):
                        adj[w] &= ~(1 << u)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                continue

        # 3. Greedy maximization
        # Try to add all possible edges
        # Sort pairs by potential to form triangles? 
        # Just iterate through all pairs multiple times
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_triangle_free(adj, u, v):
                            add_edge_if_valid(adj, u, v)
                            changed = True
                            
        # 4. Check 3-colorability
        if not is_3_colorable(adj):
            m = count_edges(adj)
            if m > best_edges:
                best_edges = m
                best_graph = get_edges(adj)
                
    return best_graph

