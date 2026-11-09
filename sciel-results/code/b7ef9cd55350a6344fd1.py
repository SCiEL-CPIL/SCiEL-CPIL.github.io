"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_path = []
    best_len = 0
    
    # Helper: count reachable unvisited vertices from a given vertex
    def count_reachable(start_v, visited):
        """Count how many unvisited vertices are reachable from start_v."""
        if start_v in visited:
            return 0
        seen = set()
        stack = [start_v]
        while stack:
            v = stack.pop()
            if v in seen or v in visited:
                continue
            seen.add(v)
            for nb in adj[v]:
                if nb not in seen and nb not in visited:
                    stack.append(nb)
        return len(seen)
    
    # DFS with backtracking and pruning
    def dfs(path, visited, best_len_ref):
        nonlocal best_path, best_len
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best_path = list(path)
        
        if time.time() > deadline:
            return
        
        # Pruning: if current path + reachable unvisited < best_len, prune
        last = path[-1]
        reachable = count_reachable(last, visited)
        if current_len + reachable <= best_len:
            return
        
        # Try neighbors in randomized order
        neighbors = list(adj[last] - visited)
        # Sort by degree (prefer lower degree first - heuristic for finding longer paths)
        neighbors.sort(key=lambda x: (len(adj[x] - visited), x))
        
        # Randomize slightly for diversity
        if len(neighbors) > 1:
            # Try a random shuffle with some bias toward low-degree
            rng = random.Random(hash((tuple(path), len(neighbors))))
            # Partial shuffle: swap some elements
            for i in range(len(neighbors) - 1, 0, -1):
                if rng.random() < 0.3:
                    j = rng.randint(0, i)
                    neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
        
        for nb in neighbors:
            path.append(nb)
            visited.add(nb)
            dfs(path, visited, best_len)
            path.pop()
            visited.remove(nb)
            if time.time() > deadline:
                return
    
    # Try all starting vertices
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        visited = {start}
        dfs(path, visited, best_len)
    
    # If no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path

# EVOLVE-BLOCK-END
