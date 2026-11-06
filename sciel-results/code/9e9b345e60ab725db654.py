import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    
    # Build Grötzsch graph on vertices 0-10
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
        
    # Shadows
    cycle_edges = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle_edges:
        su = 5 + u
        sv = 5 + v
        add_edge(su, v)
        add_edge(sv, u)
        
    # Top 10 connected to all shadows
    for i in range(5):
        add_edge(10, 5 + i)
        
    initial_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                initial_edges.append([i, j])
                
    best_edges = initial_edges[:]
    best_count = len(initial_edges)
    
    # Potential edges (non-edges in base graph)
    potential_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                potential_edges.append((i, j))
                
    def init_masks():
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
        return masks
        
    base_masks = init_masks()
    
    def creates_triangle_mask(u, v, masks):
        return (masks[u] & masks[v]) != 0
    
    def is_3_colorable(masks):
        """Check if graph represented by masks is 3-colorable using backtracking."""
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda v: bin(masks[v]).count('1'), reverse=True)
        
        # color[v] = 0 (unassigned), 1, 2, or 3
        color = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Find forbidden colors from neighbors
            forbidden = 0
            for nb in range(n):
                if (masks[v] >> nb) & 1 and color[nb]:
                    forbidden |= (1 << color[nb])
            
            # Try colors 1, 2, 3
            for c in [1, 2, 3]:
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = 0
            
            return False
        
        return backtrack(0)
    
    num_iterations = 10000
    iter_count = 0
    
    while time.time() < deadline and iter_count < num_iterations:
        iter_count += 1
        
        masks = base_masks[:]
        current_edges = initial_edges[:]
        
        # Shuffle potentials
        shuffled = potential_edges[:]
        random.shuffle(shuffled)
        
        added = 0
        for u, v in shuffled:
            if not creates_triangle_mask(u, v, masks):
                # Tentatively add edge
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append([u, v])
                added += 1
        
        # Check if the resulting graph is NOT 3-colorable
        if not is_3_colorable(masks):
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges[:]
        else:
            # Graph is 3-colorable, need to find a minimal non-3-colorable subset
            # or try a different approach: remove some edges to break 3-colorability
            # This is complex, so we just skip this result
            pass
    
    return best_edges
