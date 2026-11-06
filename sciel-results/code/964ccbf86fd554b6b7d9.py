import numpy as np
import random

def construct():
    n = 40
    m = 401
    
    # 1. Structured Initialization: K_20,20 + 1 edge
    # This is a much better starting point than a random graph for this specific problem.
    # K_20,20 has 0 triangles. Adding 1 edge inside creates 20 triangles sharing that edge.
    # Max book is 20.
    # We use Simulated Annealing to reduce this.
    
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    # Add all edges between A (0..19) and B (20..39)
    for u in range(20):
        for v in range(20, 40):
            adj[u, v] = True
            adj[v, u] = True
            edges.add((u, v))
            
    # Add 1 extra edge inside A to reach m=401
    # (0, 1)
    u, v = 0, 1
    adj[u, v] = True
    adj[v, u] = True
    edges.add((u, v))
    
    def get_max_book(adj, edges):
        max_book = 0
        for u, v in edges:
            common = np.count_nonzero(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
        
    current_max_book = get_max_book(adj, edges)
    
    # Simulated Annealing Parameters
    # We want to cool down slowly to escape local minima.
    T_start = 5.0
    T_end = 0.001
    iterations = 30000
    
    random.seed(42)
    
    for i in range(iterations):
        # Linear or exponential cooling? Exponential is standard.
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Move: Swap a random edge with a random non-edge
        # To speed up, we can maintain a list of edges and a list of non-edges?
        # Or just sample.
        
        # Pick random edge to remove
        u, v = random.choice(list(edges))
        
        # Pick random non-edge to add
        # Rejection sampling
        found = False
        while not found:
            x, y = random.sample(range(n), 2)
            if x > y:
                x, y = y, x
            if (x, y) not in edges:
                found = True
                
        # Tentatively apply swap
        adj[u, v] = False
        adj[v, u] = False
        edges.discard((u, v))
        
        adj[x, y] = True
        adj[y, x] = True
        edges.add((x, y))
        
        new_max_book = get_max_book(adj, edges)
        
        delta = new_max_book - current_max_book
        
        if delta <= 0:
            current_max_book = new_max_book
        else:
            # Accept with probability exp(-delta / T)
            # If T is small, this is unlikely.
            if random.random() < np.exp(-delta / T):
                current_max_book = new_max_book
            else:
                # Revert
                adj[x, y] = False
                adj[y, x] = False
                edges.discard((x, y))
                
                adj[u, v] = True
                adj[v, u] = True
                edges.add((u, v))
                
    result = [list(e) for e in edges]
    return result
