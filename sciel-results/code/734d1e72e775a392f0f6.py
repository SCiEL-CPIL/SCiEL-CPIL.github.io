import time
import random

def is_3_colorable(adj, n):
    """
    Returns True if the graph defined by adj (list of lists of bools) is 3-colorable.
    Uses backtracking.
    """
    colors = [-1] * n
    
    # Order vertices by degree descending to prune earlier
    order = list(range(n))
    order.sort(key=lambda x: sum(adj[x]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Available colors
        used = [False] * 3
        for u in range(n):
            if adj[v][u] and colors[u] != -1:
                used[colors[u]] = True
        
        for c in range(3):
            if not used[c]:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    n = 16
    # Initialize with Grötzsch graph on 0..10
    # Edges of Grötzsch graph:
    # Cycle 0-1-2-3-4-0
    # 5-6-7-8-9-5
    # 0-5, 1-6, 2-7, 3-8, 4-9
    # 10 connected to 5,6,7,8,9
    # Wait, standard Grötzsch construction:
    # V = {0..4} U {5..9} U {10}
    # Edges:
    # (i, i+1 mod 5) for i in 0..4
    # (i, 5+i) for i in 0..4  (Verticals? No, usually copies)
    # Let's use a known list or construct carefully.
    # Standard:
    # 0-1, 1-2, 2-3, 3-4, 4-0
    # 5-6, 6-7, 7-8, 8-9, 9-5
    # 0-5, 1-6, 2-7, 3-8, 4-9
    # 5-10, 6-10, 7-10, 8-10, 9-10
    
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,6), (6,7), (7,8), (8,9), (9,5),
        (0,5), (1,6), (2,7), (3,8), (4,9),
        (5,10), (6,10), (7,10), (8,10), (9,10)
    ]
    
    adj = [[False]*n for _ in range(n)]
    for u, v in grotzsch_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Vertices 11-15 start isolated
    
    def get_edge_count():
        return sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
    
    def can_add_edge(u, v):
        if adj[u][v]:
            return False
        # Check triangle
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return False
        return True

    # Phase 1: Add edges greedily, checking 3-colorability only when we think we are close or periodically
    # To ensure 4-chromaticity, we will verify at the end of each "phase" or every few steps.
    # Actually, let's just do local search with full verification on acceptance.
    # Since n=16, verification is fast.
    
    current_edges = get_edge_count()
    
    # Helper to try adding a random edge
    def try_add_random_edge():
        nonlocal current_edges
        # Find all addable edges (no triangle)
        addable = []
        for u in range(n):
            for v in range(u+1, n):
                if can_add_edge(u, v):
                    addable.append((u, v))
        
        if not addable:
            return False
        
        u, v = random.choice(addable)
        # Tentatively add
        adj[u][v] = True
        adj[v][u] = True
        
        # Check 3-colorability
        if not is_3_colorable(adj, n):
            # Accept
            current_edges += 1
            return True
        else:
            # Reject
            adj[u][v] = False
            adj[v][u] = False
            return False

    # Helper to try swapping: remove one edge, add one or more
    def try_swap():
        nonlocal current_edges
        # Remove a random edge
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        if not all_edges:
            return False
            
        u, v = random.choice(all_edges)
        adj[u][v] = False
        adj[v][u] = False
        current_edges -= 1
        
        # Try to add up to 2 edges
        # Simple: try to add one random addable edge
        # If we add 1, net change 0. If we add 2, net change +1.
        # We just want to escape local minima.
        
        # Find addable edges
        addable = []
        for i in range(n):
            for j in range(i+1, n):
                if can_add_edge(i, j):
                    addable.append((i, j))
        
        if not addable:
            # Revert
            adj[u][v] = True
            adj[v][u] = True
            current_edges += 1
            return False
            
        # Pick a random addable edge
        nu, nv = random.choice(addable)
        adj[nu][nv] = True
        adj[nv][nu] = True
        current_edges += 1
        
        # Check 3-colorability
        if not is_3_colorable(adj, n):
            # Accept swap (net 0 change in edges, but different structure)
            # We might want to try to add another edge if possible?
            # For now, accept the swap.
            return True
        else:
            # Revert
            adj[nu][nv] = False
            adj[nv][nu] = False
            current_edges -= 1
            adj[u][v] = True
            adj[v][u] = True
            current_edges += 1
            return False

    # Main Loop
    no_improvement_count = 0
    last_best = current_edges
    
    while time.time() < deadline:
        # Try to add an edge
        if try_add_random_edge():
            no_improvement_count = 0
            last_best = current_edges
        else:
            no_improvement_count += 1
            
            # If stuck for a while, try swaps
            if no_improvement_count > 50:
                if try_swap():
                    no_improvement_count = 0
                else:
                    # If swap also fails or doesn't help, maybe restart part of the graph?
                    # Or just keep trying.
                    pass

    # Final Verification
    # The loop ensures we only accept moves that keep the graph 4-chromatic (not 3-colorable).
    # However, the initial state (Grötzsch) is 4-chromatic.
    # So the final state should be 4-chromatic.
    
    # But what if the random choices led to a state where no more edges can be added,
    # but the graph is 3-colorable?
    # The `try_add_random_edge` rejects moves that make it 3-colorable.
    # The `try_swap` rejects moves that make it 3-colorable.
    # So the invariant "Graph is 4-chromatic" is maintained.
    
    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
