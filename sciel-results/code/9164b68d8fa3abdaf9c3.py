import random
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Function to check if a subset of vertices induces a regular graph
    def is_regular_subset(vertices, adj):
        if len(vertices) < 6:
            return False
        k = len(vertices)
        degrees = []
        for u in vertices:
            deg = 0
            for v in vertices:
                if u != v:
                    # Check if edge (u, v) exists
                    # adj[u] is a bitmask where bit v is 1 if edge exists
                    if (adj[u] >> v) & 1:
                        deg += 1
            degrees.append(deg)
        # Check if all degrees are equal
        if all(d == degrees[0] for d in degrees):
            return True
        return False

    # Function to check if the whole graph is valid
    def is_valid_graph(n, adj):
        # Check all subsets of size >= 6
        # Optimization: If n < 6, it's valid.
        if n < 6:
            return True
        
        # Iterate over all masks? No, iterate over combinations.
        # For n=20, combinations of 6 is 38760.
        # Combinations of 7 is 77520.
        # Total is manageable.
        
        from itertools import combinations
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset(subset, adj):
                    return False
        return True

    # Search for a valid graph
    # We try sizes from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a valid graph of size n
        found = False
        best_graph = None
        
        # Strategy: Random initialization + Local Search
        # We will try a few random graphs and then mutate.
        
        for attempt in range(1000):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check if valid
            if is_valid_graph(n, adj):
                found = True
                best_graph = adj
                break
            
            # If not valid, try to repair
            # Get a violating subset
            # This part is expensive to find every time.
            # Instead, just restart with a new random graph?
            # Or use a heuristic: flip random edges.
            
            # Let's try a few mutations
            current_adj = adj[:]
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Flip a random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                
                if is_valid_graph(n, current_adj):
                    found = True
                    best_graph = current_adj
                    break
            
            if found:
                break
        
        if found:
            # Convert adj to edges list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (best_graph[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # If n=20 failed, try n=19, etc.
        
    # Fallback: n=6
    # A non-regular graph on 6 vertices
    # e.g., K6 minus an edge
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1): # Missing edge (0,1)
                edges.append([i, j])
    return {'n': n, 'edges': edges}
