"""Long path finder using Greedy Construction and Local Search."""

import time

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    # Pre-compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Heuristic: Start from low degree nodes
    start_candidates = sorted(range(n), key=lambda x: (degrees[x], x))
    
    best_path = []
    best_len = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    # Greedy construction function
    def greedy_construct(start_node):
        path = [start_node]
        visited = {start_node}
        
        while len(path) < n:
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited]
            if not neighbors:
                break
            
            # Warnsdorff's rule: pick neighbor with fewest unvisited neighbors
            neighbors.sort(key=lambda v: len(adj[v] - visited))
            next_node = neighbors[0]
            
            path.append(next_node)
            visited.add(next_node)
            
        return path

    # Local search: Try to insert unvisited nodes into the path
    def local_search(path):
        visited = set(path)
        unvisited = [v for v in range(n) if v not in visited]
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for u in unvisited[:]:
                # Try to insert u between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    if u in adj[path[i]] and u in adj[path[i+1]]:
                        new_path = path[:i+1] + [u] + path[i+1:]
                        path = new_path
                        visited.add(u)
                        unvisited.remove(u)
                        improved = True
                        break
                if improved:
                    break
        
        return path

    # Try multiple starts
    for s in start_candidates:
        if time.time() - start_time > time_limit:
            break
        
        path = greedy_construct(s)
        
        # Apply local search
        path = local_search(path)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
        if best_len == n:
            break
            
    return best_path
