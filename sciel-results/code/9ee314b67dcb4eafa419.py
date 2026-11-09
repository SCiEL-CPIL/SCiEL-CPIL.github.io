import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to sorted lists for faster iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    deadline = time.time() + 55  # Safety margin before 60s kill
    
    # Precompute degree for heuristics
    deg = [len(adj_list[i]) for i in range(n)]
    
    # Sort vertices by degree (high degree first for starting points)
    vertices_by_degree = sorted(range(n), key=lambda x: (-deg[x], x))
    
    def dfs(path, visited):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        current = path[-1]
        # Pruning: if current length + remaining unvisited <= best_len, no point continuing
        # But we need a tighter bound: count reachable unvisited from current
        # Simple bound: len(path) + (n - len(visited)) <= best_len
        if len(path) + (n - len(visited)) <= best_len:
            return
        
        # Get unvisited neighbors
        unvisited_neighbors = [v for v in adj_list[current] if v not in visited]
        
        if not unvisited_neighbors:
            return
        
        # Order neighbors: prefer those with fewer unvisited options (dead-ends first)
        # This helps find long paths faster by exploring constrained branches
        unvisited_neighbors.sort(key=lambda v: (len([w for w in adj_list[v] if w not in visited]), v))
        
        for v in unvisited_neighbors:
            if time.time() > deadline:
                return
            path.append(v)
            visited.add(v)
            dfs(path, visited)
            path.pop()
            visited.remove(v)
    
    # Try multiple starting vertices with different random orderings
    rng = random.Random(42)
    
    # First pass: deterministic, try all vertices in degree order
    for start in vertices_by_degree:
        if time.time() > deadline:
            break
        path = [start]
        visited = {start}
        dfs(path, visited)
    
    # Second pass: randomized restarts to escape local optima
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random start vertex
        start = rng.randint(0, n - 1)
        path = [start]
        visited = {start}
        
        # Randomized DFS with some shuffling
        def dfs_random(path, visited):
            nonlocal best_path, best_len
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            current = path[-1]
            if len(path) + (n - len(visited)) <= best_len:
                return
            
            unvisited_neighbors = [v for v in adj_list[current] if v not in visited]
            if not unvisited_neighbors:
                return
            
            # Shuffle neighbors for randomization
            rng.shuffle(unvisited_neighbors)
            # But still bias toward constrained nodes
            unvisited_neighbors.sort(key=lambda v: (len([w for w in adj_list[v] if w not in visited]), v))
            # Partially preserve order by only shuffling top few
            if len(unvisited_neighbors) > 3:
                # Shuffle first 3 for diversity
                unvisited_neighbors[:3] = rng.sample(unvisited_neighbors[:3], 3)
            
            for v in unvisited_neighbors:
                if time.time() > deadline:
                    return
                path.append(v)
                visited.add(v)
                dfs_random(path, visited)
                path.pop()
                visited.remove(v)
        
        dfs_random(path, visited)
    
    # Third pass: targeted search from high-degree vertices with more iterations
    for start in vertices_by_degree[:20]:
        if time.time() > deadline:
            break
        for trial in range(10):
            if time.time() > deadline:
                break
            path = [start]
            visited = {start}
            
            def dfs_targeted(path, visited):
                nonlocal best_path, best_len
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                
                current = path[-1]
                if len(path) + (n - len(visited)) <= best_len:
                    return
                
                unvisited_neighbors = [v for v in adj_list[current] if v not in visited]
                if not unvisited_neighbors:
                    return
                
                # Sort by: prefer neighbors that lead to more unvisited vertices
                unvisited_neighbors.sort(key=lambda v: (-len([w for w in adj_list[v] if w not in visited]), v))
                
                for v in unvisited_neighbors:
                    if time.time() > deadline:
                        return
                    path.append(v)
                    visited.add(v)
                    dfs_targeted(path, visited)
                    path.pop()
                    visited.remove(v)
            
            dfs_targeted(path, visited)
    
    return best_path
# EVOLVE-BLOCK-END
