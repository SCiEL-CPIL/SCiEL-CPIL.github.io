import random
import time

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_path = []
    best_len = 0
    
    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
    # 1. Deterministic Greedy from all nodes
    for start in range(n):
        path = [start]
        seen = {start}
        while adj[path[-1]] - seen:
            candidates = adj[path[-1]] - seen
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        update_best(path)
        
    # If we found a Hamiltonian path, return it immediately
    if best_len == n:
        return best_path

    # 2. Randomized Greedy with Local Search
    rng = random.Random(748) # Use task seed for reproducibility of the search process
    
    start_time = time.time()
    time_limit = 85.0 # 5s safety margin
    
    while time.time() - start_time < time_limit:
        if best_len == n:
            break
            
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        
        # Randomized Greedy Construction
        while adj[path[-1]] - seen:
            if len(path) == n:
                break
            candidates = list(adj[path[-1]] - seen)
            candidates.sort(key=lambda w: (len(adj[w] - seen), w))
            
            r = rng.random()
            if r < 0.8 and len(candidates) > 0:
                v = candidates[0]
            elif r < 0.95 and len(candidates) > 1:
                v = candidates[1]
            else:
                k = min(5, len(candidates))
                v = rng.choice(candidates[:k])
                
            path.append(v)
            seen.add(v)
            
        # Local Search: Insertion
        improved = True
        while improved:
            if len(path) == n:
                break
            improved = False
            unused = set(range(n)) - seen
            
            # Try inserting in the middle
            # Iterate from end to start to avoid index shifting issues if we inserted, 
            # but we break after insertion so it doesn't matter.
            for i in range(len(path) - 1):
                u1 = path[i]
                u2 = path[i+1]
                # Common neighbors of u1 and u2 that are unused
                common = adj[u1] & adj[u2] & unused
                if common:
                    # Pick a vertex to insert
                    # Heuristic: pick one with smallest degree to preserve options? 
                    # Or random. Random is safer for diversification.
                    u = rng.choice(list(common))
                    path.insert(i+1, u)
                    seen.add(u)
                    improved = True
                    break # Restart loop
            
            # If no middle insertion, try ends
            if not improved and unused:
                # Try start
                if path:
                    end_v = path[0]
                    candidates = adj[end_v] & unused
                    if candidates:
                        u = rng.choice(list(candidates))
                        path.insert(0, u)
                        seen.add(u)
                        improved = True
                # Try end
                if not improved and path:
                    end_v = path[-1]
                    candidates = adj[end_v] & unused
                    if candidates:
                        u = rng.choice(list(candidates))
                        path.append(u)
                        seen.add(u)
                        improved = True
                        
        update_best(path)

    if not best_path:
        return [0]
        
    return best_path

# EVOLVE-BLOCK-END
