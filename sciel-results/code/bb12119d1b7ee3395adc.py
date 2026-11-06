import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = len(best_edges)
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def check_valid(adj, neigh, u, v, c):
        # x is neighbour of v
        for x in neigh[v]:
            if x == u:
                continue
            cvx = adj[v][x]
            if cvx == c:
                continue
            
            # y is neighbour of u
            for y in neigh[u]:
                if y == v or y == x:
                    continue
                if adj[x][y] == -1:
                    continue
                
                cyu = adj[y][u]
                cxy = adj[x][y]
                
                # Check distinctness
                if c == cxy or c == cyu or cvx == cxy or cvx == cyu or cxy == cyu:
                    return False
        return True

    for attempt in range(500):
        if time.time() > deadline:
            break
        
        adj = [[-1]*n for _ in range(n)]
        neigh = [[] for _ in range(n)]
        edges = []
        
        strategy = attempt % 4
        
        if strategy == 0:
            order = all_pairs[:]
            random.shuffle(order)
        elif strategy == 1:
            # Cycle first
            order = [(i, (i+1)%n) for i in range(n)]
            remaining = [p for p in all_pairs if p not in order]
            random.shuffle(remaining)
            order.extend(remaining)
        elif strategy == 2:
            # Bipartite first
            order = [(u, v) for u in range(10) for v in range(10, 20)]
            remaining = [p for p in all_pairs if p not in order]
            random.shuffle(remaining)
            order.extend(remaining)
        else:
            order = all_pairs[:]
            random.shuffle(order)

        for u, v in order:
            if time.time() > deadline:
                break
            
            # Try colours
            # Heuristic: colours least used in neigh[u] and neigh[v]
            # Simple: random order
            colours = list(range(n))
            random.shuffle(colours)
            
            added = False
            for c in colours:
                if check_valid(adj, neigh, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    neigh[u].append(v)
                    neigh[v].append(u)
                    edges.append([u, v, c])
                    added = True
                    break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
