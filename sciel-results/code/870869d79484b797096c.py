"""Rewritten program: randomized backtracking search for longest simple path."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    deg = [len(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    deadline = time.time() + 85  # 85s safety margin
    
    def try_extend_forward(path, seen):
        """Greedy extension forward using Warnsdorff heuristic."""
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            # Choose neighbor with fewest unvisited neighbors (Warnsdorff)
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def try_extend_backward(path, seen):
        """Greedy extension backward using Warnsdorff heuristic."""
        while True:
            candidates = adj[path[0]] - seen
            if not candidates:
                break
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
        return path
    
    def extend_both_ends(path, seen):
        """Alternate extending both ends."""
        improved = True
        while improved:
            improved = False
            old_len = len(path)
            path = try_extend_forward(path, seen)
            if len(path) > old_len:
                improved = True
            old_len = len(path)
            path = try_extend_backward(path, seen)
            if len(path) > old_len:
                improved = True
        return path
    
    def randomized_dfs(start, max_depth, rng):
        """Randomized DFS with backtracking, limited depth."""
        path = [start]
        seen = {start}
        best_local = list(path)
        
        def dfs(current_depth, remaining):
            nonlocal best_local
            if time.time() > deadline:
                return
            if len(path) > len(best_local):
                best_local = list(path)
            
            if remaining <= 0:
                return
            
            # Get unvisited neighbors, shuffle for randomness
            candidates = list(adj[path[-1]] - seen)
            if not candidates:
                return
            
            # Sort by Warnsdorff but add some randomness
            rng.shuffle(candidates)
            # Prioritize lower-degree neighbors
            candidates.sort(key=lambda w: (len(adj[w] - seen), rng.random()))
            
            for v in candidates:
                path.append(v)
                seen.add(v)
                dfs(current_depth + 1, remaining - 1)
                path.pop()
                seen.discard(v)
                if time.time() > deadline:
                    return
        
        dfs(0, max_depth)
        return best_local
    
    # Strategy 1: Deterministic greedy from each start
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        path = extend_both_ends(path, seen)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Strategy 2: Randomized DFS with backtracking from promising starts
    # Start from low-degree vertices (more likely to be endpoints)
    starts_sorted = sorted(range(n), key=lambda i: deg[i])
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Vary the start vertex
        if attempt < 30:
            start = starts_sorted[attempt % len(starts_sorted)]
        else:
            rng = random.Random(attempt)
            start = rng.randint(0, n - 1)
        
        rng = random.Random(42 + attempt)
        
        # Vary max depth
        if attempt < 10:
            max_depth = 40
        elif attempt < 30:
            max_depth = 30
        else:
            max_depth = rng.randint(15, 35)
        
        path = randomized_dfs(start, max_depth, rng)
        
        # Try to extend the found path
        seen = set(path)
        path = extend_both_ends(path, seen)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Strategy 3: Local improvement - try to insert unvisited vertices into the path
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            seen = set(best_path)
            unvisited = [i for i in range(n) if i not in seen]
            
            for u in unvisited:
                if time.time() > deadline:
                    break
                # Try to insert u between consecutive vertices in best_path
                for i in range(len(best_path) - 1):
                    a, b = best_path[i], best_path[i + 1]
                    if u in adj[a] and u in adj[b]:
                        new_path = best_path[:i+1] + [u] + best_path[i+1:]
                        new_seen = set(new_path)
                        new_path = extend_both_ends(new_path, new_seen)
                        if len(new_path) > best_len:
                            best_len = len(new_path)
                            best_path = new_path
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return best_path
# EVOLVE-BLOCK-END
