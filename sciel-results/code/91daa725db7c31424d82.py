import time
import random

def construct():
    # 1. Construct the Mycielski graph of C5 (Groetzsch graph) on vertices 0..10
    # Vertices 0..4 form C5
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the apex
    edges = []
    # C5 edges
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Edges between original and copies
    # u in 0..4, v in 0..4
    # If (u,v) in C5, add (u, 5+v) and (v, 5+u)
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
        
    # Edges from apex 10 to copies 5..9
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Current graph has 20 edges on vertices 0..10
    # Vertices 11..15 are isolated
    
    # 2. Expand to maximum triangle-free supergraph on 16 vertices
    # We maintain a set of edges for fast lookup
    edge_set = set()
    for e in edges:
        if e[0] < e[1]:
            edge_set.add((e[0], e[1]))
        else:
            edge_set.add((e[1], e[0]))
            
    # Adjacency list for fast triangle check
    adj = [set() for _ in range(16)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # List of all possible edges
    all_edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            if (i, j) not in edge_set:
                all_edges.append((i, j))
                
    # Greedily add edges that don't create triangles
    # To be deterministic and maximize, we can sort or use random order
    # Random order with multiple restarts might find better maximal subgraphs
    # But for a fixed seed, let's just do one pass or a few.
    
    def is_triangle_free_with_edge(u, v):
        # Check if adding (u,v) creates a triangle
        # Triangle exists if there is w such that (u,w) and (v,w) exist
        common = adj[u].intersection(adj[v])
        return len(common) == 0
        
    # We will add all possible edges in a specific order
    # To ensure we get a good result, we can try to add edges incident to isolated vertices first
    # or just iterate through all missing edges
    
    added = True
    # We iterate until no more edges can be added
    # Since adding an edge might block others, order matters.
    # Let's try a few random orders and keep the best.
    
    best_edges = list(edge_set)
    best_count = len(best_edges)
    
    start_time = time.time()
    deadline = start_time + 100 # 100s budget for search
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        current_edge_set = set(edge_set)
        current_adj = [set() for _ in range(16)]
        for u, v in current_edge_set:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        missing = []
        for i in range(16):
            for j in range(i + 1, 16):
                if (i, j) not in current_edge_set:
                    missing.append((i, j))
                    
        random.shuffle(missing)
        
        for u, v in missing:
            # Check if we can add (u, v)
            common = current_adj[u].intersection(current_adj[v])
            if len(common) == 0:
                current_edge_set.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        if len(current_edge_set) > best_count:
            best_count = len(current_edge_set)
            best_edges = list(current_edge_set)
            
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
