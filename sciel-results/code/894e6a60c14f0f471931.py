import random
import time
import copy

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, c, u, v):
        # Check if adding (u, v) is valid
        # 1. c[u][v] <= 2
        if c[u][v] > 2:
            return False
        
        # 2. For all w in N(u), c[w][v] <= 1
        #    Because adding (u,v) makes u a neighbor of v.
        #    If w is neighbor of u, u is a common neighbor of w and v.
        #    So c[w][v] increases by 1. Must stay <= 2.
        for w in adj[u]:
            if c[w][v] > 1:
                return False
        
        # 3. For all w in N(v), c[u][w] <= 1
        for w in adj[v]:
            if c[u][w] > 1:
                return False
        
        return True

    def greedy_construct(seed_edges=None):
        # If seed_edges is provided, start from there
        # adj[u] is set of neighbors
        # c[u][v] is count of common neighbors
        if seed_edges:
            current_edges = list(seed_edges)
            current_adj = [set() for _ in range(n)]
            current_c = [[0] * n for _ in range(n)]
            for u, v in current_edges:
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_c[u][v] += 1
                current_c[v][u] += 1
                for w in current_adj[u]:
                    current_c[w][v] += 1
                    current_c[v][w] += 1
                for w in current_adj[v]:
                    current_c[u][w] += 1
                    current_c[w][u] += 1
        else:
            current_edges = []
            current_adj = [set() for _ in range(n)]
            current_c = [[0] * n for _ in range(n)]
        
        # Get list of edges NOT in current_edges
        existing_set = set(map(tuple, current_edges))
        missing_edges = [e for e in all_edges if e not in existing_set]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, current_c, u, v):
                current_edges.append([u, v])
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_c[u][v] += 1
                current_c[v][u] += 1
                for w in current_adj[u]:
                    current_c[w][v] += 1
                    current_c[v][w] += 1
                for w in current_adj[v]:
                    current_c[u][w] += 1
                    current_c[w][u] += 1
        
        return current_edges

    def local_search(edges):
        # Try to improve a given graph
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        # Build state from best_local_edges
        def build_state(edges):
            adj = [set() for _ in range(n)]
            c = [[0] * n for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                c[u][v] += 1
                c[v][u] += 1
                for w in adj[u]:
                    c[w][v] += 1
                    c[v][w] += 1
                for w in adj[v]:
                    c[u][w] += 1
                    c[w][u] += 1
            return adj, c

        # We will modify a copy
        current_edges = list(edges)
        current_adj, current_c = build_state(current_edges)
        
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to add any edge
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_addition(current_adj, current_c, u, v):
                    current_edges.append([u, v])
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_c[u][v] += 1
                    current_c[v][u] += 1
                    for w in current_adj[u]:
                        current_c[w][v] += 1
                        current_c[v][w] += 1
                    for w in current_adj[v]:
                        current_c[u][w] += 1
                        current_c[w][u] += 1
                    improved = True
                    break
            
            if improved:
                if len(current_edges) > best_local_count:
                    best_local_count = len(current_edges)
                    best_local_edges = list(current_edges)
                continue
            
            # If no edge added, try a swap: remove one, add others
            if current_edges:
                # Pick a random edge to remove
                idx = random.randint(0, len(current_edges) - 1)
                u, v = current_edges[idx]
                
                # Save state
                saved_edges = list(current_edges)
                saved_adj = [set(s) for s in current_adj]
                saved_c = [row[:] for row in current_c]
                
                # Remove edge
                current_edges.pop(idx)
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                current_c[u][v] -= 1
                current_c[v][u] -= 1
                for w in current_adj[u]:
                    current_c[w][v] -= 1
                    current_c[v][w] -= 1
                for w in current_adj[v]:
                    current_c[u][w] -= 1
                    current_c[w][u] -= 1
                
                # Try to add edges
                missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
                random.shuffle(missing)
                for a, b in missing:
                    if is_valid_addition(current_adj, current_c, a, b):
                        current_edges.append([a, b])
                        current_adj[a].add(b)
                        current_adj[b].add(a)
                        current_c[a][b] += 1
                        current_c[b][a] += 1
                        for w in current_adj[a]:
                            current_c[w][b] += 1
                            current_c[b][w] += 1
                        for w in current_adj[b]:
                            current_c[a][w] += 1
                            current_c[w][a] += 1
                
                if len(current_edges) > best_local_count:
                    best_local_count = len(current_edges)
                    best_local_edges = list(current_edges)
                else:
                    # Revert
                    current_edges = saved_edges
                    current_adj = saved_adj
                    current_c = saved_c
        
        return best_local_edges

    # Main loop
    iterations = 0
    last_local_search = 0
    
    while time.time() < deadline:
        # Run a greedy construction
        edges = greedy_construct()
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        
        iterations += 1
        
        # Every 500 iterations, run local search on the best found so far
        if iterations % 500 == 0 and time.time() - last_local_search > 5.0:
            last_local_search = time.time()
            improved = local_search(best_edges)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved

    return best_edges
