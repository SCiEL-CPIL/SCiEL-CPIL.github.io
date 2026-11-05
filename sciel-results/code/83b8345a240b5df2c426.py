import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + 1 edge (0,1)
    adj = np.zeros((n, n), dtype=np.int8)
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
            adj[a, b] = 1
            adj[b, a] = 1
    edges.add((0, 1))
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def get_max_book(adj, edges):
        T_all = adj @ adj
        max_b = 0
        for u, v in edges:
            b = T_all[u, v]
            if b > max_b:
                max_b = b
        return max_b

    current_max_book = get_max_book(adj, edges)
    best_edges = set(edges)
    best_max_book = current_max_book
    
    # Local Search
    no_improve_count = 0
    while time.time() < deadline:
        improved = False
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if time.time() > deadline:
                break
            
            # Try removing (u,v) and adding random (x,y)
            # Increase number of random tries if stuck?
            tries = 10 + no_improve_count
            for _ in range(tries):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                
                if (x, y) in edges:
                    continue
                
                # Perform swap
                adj[u, v] = 0
                adj[v, u] = 0
                edges.remove((u, v))
                
                adj[x, y] = 1
                adj[y, x] = 1
                edges.add((x, y))
                
                new_max_book = get_max_book(adj, edges)
                
                if new_max_book < current_max_book:
                    current_max_book = new_max_book
                    improved = True
                    no_improve_count = 0
                    if current_max_book < best_max_book:
                        best_max_book = current_max_book
                        best_edges = set(edges)
                    break # Restart outer loop iteration with new state
                
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add((u, v))
                
                adj[x, y] = 0
                adj[y, x] = 0
                edges.remove((x, y))
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                # Try a random restart?
                # Just reset no_improve_count to keep trying
                no_improve_count = 0

    return [[u, v] for u, v in best_edges]
