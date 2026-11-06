"""Rewritten program: Warnsdorff-guided DFS + backtracking + path extension."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best_path = []
    
    def try_extend(path, adj):
        """Try to extend path at both ends."""
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try to extend at the front
            front = path[0]
            for v in sorted(adj[front]):
                if v not in set(path):
                    path.insert(0, v)
                    changed = True
                    break
            # Try to extend at the back
            if not changed:
                back = path[-1]
                for v in sorted(adj[back]):
                    if v not in set(path):
                        path.append(v)
                        changed = True
                        break
        return path
    
    def warnsdorff_dfs(start, adj, deadline, max_depth=200):
        """DFS with Warnsdorff heuristic and backtracking."""
        best = [start]
        path = [start]
        visited = {start}
        
        def dfs():
            nonlocal best, path, visited
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = list(path)
            
            current = path[-1]
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                return
            
            # Warnsdorff: prefer vertices with fewest unvisited neighbors
            def key(v):
                return (len(adj[v] - visited), v)
            candidates.sort(key=key)
            
            for v in candidates:
                if time.time() > deadline:
                    return
                path.append(v)
                visited.add(v)
                dfs()
                path.pop()
                visited.discard(v)
        
        dfs()
        return best
    
    # Try from each vertex as start
    # Use a randomized order to diversify
    starts = list(range(n))
    rng = random.Random(42)
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        if len(adj[start]) == 0:
            continue
        path = warnsdorff_dfs(start, adj, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Try to extend the best path
    if best_path:
        best_path = try_extend(best_path, adj)
    
    # If still empty (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
