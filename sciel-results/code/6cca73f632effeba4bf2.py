import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    def greedy_warnsdorff(start, randomize, rng=None):
        path = [start]
        visited = {start}
        while True:
            current = path[-1]
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Score neighbors: fewer unvisited neighbors is better (Warnsdorff)
            # Tie-breaker: node id
            scored = []
            for n in neighbors:
                unvisited_count = len(adj[n] - visited)
                scored.append((unvisited_count, n))
            
            scored.sort()
            
            if randomize and rng:
                # Pick from top 3 to add diversity
                top_k = scored[:3]
                choice = rng.choice(top_k)
                next_node = choice[1]
            else:
                next_node = scored[0][1]
                
            path.append(next_node)
            visited.add(next_node)
            
        return path

    def extend_path(path):
        if not path:
            return path
        visited = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            start_node = path[0]
            neighbors = [n for n in adj[start_node] if n not in visited]
            if neighbors:
                scored = [(len(adj[n] - visited), n) for n in neighbors]
                scored.sort()
                next_node = scored[0][1]
                path.insert(0, next_node)
                visited.add(next_node)
                changed = True
            
            # Extend end
            end_node = path[-1]
            neighbors = [n for n in adj[end_node] if n not in visited]
            if neighbors:
                scored = [(len(adj[n] - visited), n) for n in neighbors]
                scored.sort()
                next_node = scored[0][1]
                path.append(next_node)
                visited.add(next_node)
                changed = True
        return path

    best_path = []
    deadline = time.time() + 55.0
    
    # Phase 1: Deterministic from all starts
    for start in range(n):
        path = greedy_warnsdorff(start, False)
        path = extend_path(path)
        if len(path) > len(best_path):
            best_path = path
        if time.time() > deadline:
            break

    # Phase 2: Randomized restarts
    rng = random.Random(42)
    while time.time() < deadline:
        start = rng.randint(0, n - 1)
        path = greedy_warnsdorff(start, True, rng)
        path = extend_path(path)
        if len(path) > len(best_path):
            best_path = path
            
    return best_path
