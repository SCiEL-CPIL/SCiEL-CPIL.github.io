import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 20
    num_colors = 20
    
    def run_greedy():
        graph = [[-1 for _ in range(n)] for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        
        def is_valid_color(u, v, c):
            for x in neighbors[v]:
                if x == u: continue
                for y in neighbors[u]:
                    if y == v or y == x: continue
                    if graph[x][y] != -1:
                        c1 = c
                        c2 = graph[v][x]
                        c3 = graph[x][y]
                        c4 = graph[y][u]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
            return True

        def add_edge(u, v):
            colors = list(range(num_colors))
            random.shuffle(colors)
            for c in colors:
                if is_valid_color(u, v, c):
                    graph[u][v] = c
                    graph[v][u] = c
                    neighbors[u].append(v)
                    neighbors[v].append(u)
                    return True
            return False

        # Greedy loop
        while True:
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if graph[i][j] == -1:
                        missing.append((i, j))
            
            if not missing:
                break
                
            random.shuffle(missing)
            added = False
            for u, v in missing:
                if add_edge(u, v):
                    added = True
                    break
            
            if not added:
                break
                
        # Count edges
        edge_count = sum(len(nei) for nei in neighbors) // 2
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if graph[i][j] != -1:
                    edges.append([i, j, graph[i][j]])
        return edge_count, edges

    best_edges = 0
    best_result = []
    
    # Phase 1: Multiple restarts
    while time.time() < deadline - 10.0:
        count, edges = run_greedy()
        if count > best_edges:
            best_edges = count
            best_result = edges
            
    # Phase 2: Polish the best result
    if time.time() < deadline - 1.0 and best_result:
        # Reconstruct graph from best_result
        graph = [[-1 for _ in range(n)] for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        for e in best_result:
            u, v, c = e
            graph[u][v] = c
            graph[v][u] = c
            neighbors[u].append(v)
            neighbors[v].append(u)
            
        # Try to add more edges
        def is_valid_color(u, v, c):
            for x in neighbors[v]:
                if x == u: continue
                for y in neighbors[u]:
                    if y == v or y == x: continue
                    if graph[x][y] != -1:
                        c1 = c
                        c2 = graph[v][x]
                        c3 = graph[x][y]
                        c4 = graph[y][u]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
            return True

        def add_edge(u, v):
            colors = list(range(num_colors))
            random.shuffle(colors)
            for c in colors:
                if is_valid_color(u, v, c):
                    graph[u][v] = c
                    graph[v][u] = c
                    neighbors[u].append(v)
                    neighbors[v].append(u)
                    return True
            return False

        while time.time() < deadline - 1.0:
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if graph[i][j] == -1:
                        missing.append((i, j))
            
            if not missing:
                break
                
            random.shuffle(missing)
            added = False
            for u, v in missing:
                if add_edge(u, v):
                    added = True
                    break
            
            if not added:
                break
        
        # Update best if improved
        current_count = sum(len(nei) for nei in neighbors) // 2
        if current_count > best_edges:
            best_edges = current_count
            best_result = []
            for i in range(n):
                for j in range(i+1, n):
                    if graph[i][j] != -1:
                        best_result.append([i, j, graph[i][j]])

    return best_result
