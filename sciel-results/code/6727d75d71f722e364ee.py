import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def is_valid(coloring, k):
        # Check for bicolored cycles
        # For each pair of colors, check if induced subgraph is a forest
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # Union-Find to detect cycle
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if not union(u, v):
                            return False
        return True

    def count_bicolored_cycles(coloring, k):
        count = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if not union(u, v):
                            count += 1
                            break
        return count

    start_time = time.time()
    deadline = start_time + 80.0
    
    best_coloring = None
    best_k = n # fallback
    
    for k in range(3, n + 1):
        # Try to find a valid coloring with k colors
        # Use local search
        # Initialize randomly
        coloring = [random.randint(0, k - 1) for _ in range(n)]
        
        # Simple hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_cost = count_bicolored_cycles(coloring, k)
            if current_cost == 0:
                best_coloring = coloring[:]
                best_k = k
                break
            
            # Try to change color of a random vertex
            # Or try all vertices? n=50 is small.
            # Let's try all vertices and pick the best move
            best_move = None
            best_move_cost = current_cost
            for v in range(n):
                old_color = coloring[v]
                for new_color in range(k):
                    if new_color == old_color: continue
                    coloring[v] = new_color
                    new_cost = count_bicolored_cycles(coloring, k)
                    if new_cost < best_move_cost:
                        best_move_cost = new_cost
                        best_move = (v, new_color)
                coloring[v] = old_color
            
            if best_move:
                v, c = best_move
                coloring[v] = c
                improved = True
            else:
                # Stuck, try random restart or perturbation
                # Randomly change a few vertices
                for _ in range(5):
                    v = random.randint(0, n - 1)
                    coloring[v] = random.randint(0, k - 1)
        
        if time.time() > deadline - 5.0:
            break
            
        if best_k < k:
            break

    if best_coloring is None:
        # Fallback to trivial coloring
        return list(range(n))
    else:
        return best_coloring
