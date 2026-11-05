import time
import random
import sys

def is_3_colorable(edges, n):
    """
    Backtracking 3-coloring check.
    Returns True if the graph can be colored with 3 colors.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    colors = [-1] * n
    # Order vertices by degree descending for faster pruning
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        for c in range(3):
            conflict = False
            for u in adj[v]:
                if colors[u] == c:
                    conflict = True
                    break
            if not conflict:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def is_triangle_free(edges, n, adj):
    """
    Check if adding an edge (u, v) to the current adjacency list creates a triangle.
    We assume adj is the current adjacency list.
    """
    # A triangle is formed if u and v share a common neighbor
    if adj[u] & adj[v]:
        return False
    return True

def construct():
    n = 16
    start_time = time.time()
    max_time = 10.0 # Keep well under 120s
    
    best_edges = []
    best_count = 0
    
    # Helper to add edge to adjacency sets
    def add_edge_to_adj(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)
        
    def remove_edge_from_adj(adj, u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    # Initial population: Grötzsch graph + random completion
    # Grötzsch graph edges (11 vertices)
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Cycle 5
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), # Connections to 5-9
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), # Connections to 10
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5) # Cycle 5-9? No, Mycielski adds edges between new nodes?
        # Let's use a standard definition or just generate a random valid one.
        # Actually, let's just start with a random dense triangle-free graph and repair/optimize.
    ]
    
    # Strategy: 
    # 1. Generate a random triangle-free graph.
    # 2. Check if it's 4-chromatic. If not, try to modify it.
    # 3. Use local search to maximize edges while maintaining 4-chromaticity.
    
    # To ensure we start with a 4-chromatic graph, we can use the Grötzsch graph as a core.
    # Vertices 0-10 form Grötzsch. 11-15 are isolated.
    # We will try to connect 11-15 to the rest and add edges between 0-10 if possible (though Grötzsch is maximal triangle-free on 11? No, max is floor(11^2/4)=30, Grötzsch has 20).
    
    # Let's define Grötzsch edges properly.
    # Mycielski of C5:
    # Original C5: 0-1-2-3-4-0
    # New vertices: 5,6,7,8,9 (copies of 0..4) and 10 (root)
    # Edges:
    # 1. Original C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # 2. Copies connected to neighbors of original:
    #    5 (copy of 0) connected to 1, 4
    #    6 (copy of 1) connected to 0, 2
    #    7 (copy of 2) connected to 1, 3
    #    8 (copy of 3) connected to 2, 4
    #    9 (copy of 4) connected to 3, 0
    # 3. Root 10 connected to all copies 5..9
    # Total edges: 5 + 10 + 5 = 20.
    
    init_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # This graph is triangle-free and 4-chromatic.
    # We start with this and try to add edges involving vertices 11-15 and potentially add more edges among 0-10 if we can (but Grötzsch is not maximal, we can add edges? 
    # Wait, Grötzsch is triangle-free. Can we add edges to it without creating triangles?
    # Yes, likely. For example, is (0,2) present? No. Do 0 and 2 share a neighbor? 
    # N(0)={1,4,6,9}. N(2)={1,3,7,8}. Intersection {1}. So (0,2) creates triangle 0-1-2.
    # So we can't just add any edge.
    
    # Local Search Loop
    best_adj = [set() for _ in range(n)]
    best_edge_set = set()
    
    def update_best(edges_set):
        nonlocal best_edges, best_count
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = list(edges_set)
            
    # Initialize best with the Grötzsch graph
    for u, v in init_edges:
        best_edge_set.add((min(u,v), max(u,v)))
    for u, v in best_edge_set:
        best_adj[u].add(v)
        best_adj[v].add(u)
    update_best(best_edge_set)
    
    # All possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated Annealing / Randomized Local Search
    # We will try to add edges. If we get stuck, we remove a random edge to escape.
    
    current_edges = set(best_edge_set)
    current_adj = [set() for _ in range(n)]
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
        
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # 1. Try to add a random missing edge
        missing_edges = [e for e in all_pairs if e not in current_edges]
        if missing_edges:
            u, v = random.choice(missing_edges)
            # Check triangle-free
            if not (current_adj[u] & current_adj[v]):
                # Check 3-colorability
                # Tentatively add
                current_edges.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
                if not is_3_colorable(list(current_edges), n):
                    # Valid move, accept
                    update_best(current_edges)
                else:
                    # Revert
                    current_edges.remove((u, v))
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    # If we frequently fail here, we might be in a local optimum
                    # where all addable edges make it 3-colorable.
                    # Strategy: Remove a random edge from current graph to "break" the coloring constraint
                    # and allow new structures.
                    if current_edges:
                        # Remove an edge that is not in the initial core? Or just random?
                        # Removing a random edge might reduce edge count, but allows re-addition of different edges.
                        rem_u, rem_v = random.choice(list(current_edges))
                        current_edges.remove((rem_u, rem_v))
                        current_adj[rem_u].remove(rem_v)
                        current_adj[rem_v].remove(rem_u)
                        # After removal, it's definitely 3-colorable or not? 
                        # Removing edges cannot make a non-3-colorable graph 3-colorable? 
                        # Wait. If G is not 3-colorable, G-e might be 3-colorable. 
                        # Yes. So we are now in a 3-colorable state.
                        # We need to get back to a non-3-colorable state.
                        # This is tricky. 
                        
                        # Better strategy: Only add edges that preserve non-3-colorability.
                        # If we are stuck, we should try a different starting point or a "swap".
                        
                        # Let's try a Swap: Remove (a,b), Add (c,d).
                        # This is complex. 
                        
                        # Alternative: If we revert, just continue trying other edges.
                        # If we fail many times in a row, restart from best.
                        
        # Occasionally restart from best to escape deep local minima
        if random.random() < 0.01:
            current_edges = set(best_edge_set)
            current_adj = [set() for _ in range(n)]
            for u, v in current_edges:
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        # If we are not improving, we can also try to remove an edge and see if we can add two others?
        # For now, simple add/remove heuristic.
        
        # To avoid infinite loops of failing to add, we track failures.
        
    return [[u, v] for u, v in best_edges]
