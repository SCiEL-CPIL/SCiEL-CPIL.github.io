import random
import time

def construct():
    N = 20
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges = 0
    best_graph = []
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    while time.time() < deadline:
        # Random restart
        random.shuffle(all_edges)
        
        # Adjacency matrix for fast lookup
        adj = [[0] * N for _ in range(N)]
        edge_count = 0
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) is valid
            valid = True
            
            # 1. Check triples (u, x, y) where v becomes a common neighbor
            # v is in N(x) and N(y) -> v in N(x) ∩ N(y)
            # We need |N(u) ∩ N(x) ∩ N(y)| <= 1 currently
            # N(u) currently does not include v.
            # So we count common neighbors of u, x, y excluding v (which is not in N(u) yet anyway).
            
            # Iterate x
            for x in range(N):
                if x == u or x == v:
                    continue
                if adj[v][x] == 0:
                    continue
                # v is neighbor of x. Check y
                for y in range(N):
                    if y == u or y == v or y == x:
                        continue
                    if adj[v][y] == 0:
                        continue
                    # v is neighbor of x and y.
                    # Check if u, x, y already have 2 common neighbors.
                    # If so, adding v to N(u) makes it 3.
                    common = 0
                    for z in range(N):
                        if z == u or z == v or z == x or z == y:
                            continue
                        if adj[u][z] and adj[x][z] and adj[y][z]:
                            common += 1
                            if common >= 2:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # 2. Check triples (v, x, y) where u becomes a common neighbor
            for x in range(N):
                if x == u or x == v:
                    continue
                if adj[u][x] == 0:
                    continue
                for y in range(N):
                    if y == u or y == v or y == x:
                        continue
                    if adj[u][y] == 0:
                        continue
                    common = 0
                    for z in range(N):
                        if z == u or z == v or z == x or z == y:
                            continue
                        if adj[v][z] and adj[x][z] and adj[y][z]:
                            common += 1
                            if common >= 2:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Add edge
                adj[u][v] = 1
                adj[v][u] = 1
                current_edges.append([u, v])
                edge_count += 1
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = current_edges[:]
            
        # Optional: Local search / Hill Climbing
        # Try to add any remaining edge that was skipped
        # This might be complex to implement correctly with the constraint check.
        # Given N=20, random restarts are likely sufficient.
        
    return best_graph
