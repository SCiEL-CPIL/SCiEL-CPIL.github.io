"""Improved longest path search with randomized backtracking and better heuristics."""

import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = [len(adj[i]) for i in range(n)]
    
    best = []
    best_len = 0
    
    # Precompute adjacency lists sorted by degree for quick access
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    def get_unvisited_neighbors(v, seen):
        """Get unvisited neighbors of v, sorted by a composite heuristic."""
        result = []
        for u in adj_list[v]:
            if u not in seen:
                # Count unvisited neighbors of u (excluding v)
                unvisited_count = sum(1 for w in adj_list[u] if w not in seen and w != v)
                # Heuristic: prefer nodes with fewer unvisited options (Warnsdorff)
                # but also consider degree to avoid isolated dead-ends
                result.append((unvisited_count, deg[u], u))
        result.sort()
        return [u for _, _, u in result]
    
    def backtrack(path, seen):
        nonlocal best, best_len
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best = path[:]
        
        current = path[-1]
        # Pruning: can we beat best?
        remaining = n - len(seen)
        if current_len + remaining <= best_len:
            return
        
        neighbors = get_unvisited_neighbors(current, seen)
        if not neighbors:
            return
        
        for v in neighbors:
            if time.time() >= deadline:
                return
            path.append(v)
            seen.add(v)
            backtrack(path, seen)
            path.pop()
            seen.discard(v)
    
    # Strategy 1: Deterministic search from each vertex
    for start in range(n):
        if time.time() >= deadline:
            break
        path = [start]
        seen = {start}
        backtrack(path, seen)
    
    # Strategy 2: Randomized restarts with shuffled neighbor order
    rng = random.Random(42)
    for restart in range(500):
        if time.time() >= deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        
        # Use a randomized variant of the heuristic
        def backtrack_rand(path, seen):
            nonlocal best, best_len
            current_len = len(path)
            if current_len > best_len:
                best_len = current_len
                best = path[:]
            
            current = path[-1]
            remaining = n - len(seen)
            if current_len + remaining <= best_len:
                return
            
            # Get unvisited neighbors with randomized tie-breaking
            neighbors = []
            for u in adj_list[current]:
                if u not in seen:
                    unvisited_count = sum(1 for w in adj_list[u] if w not in seen and w != current)
                    # Add small random perturbation for tie-breaking
                    perturbation = rng.random() * 0.1
                    neighbors.append((unvisited_count + perturbation, deg[u], u))
            neighbors.sort()
            
            for _, _, v in neighbors:
                if time.time() >= deadline:
                    return
                path.append(v)
                seen.add(v)
                backtrack_rand(path, seen)
                path.pop()
                seen.discard(v)
        
        backtrack_rand(path, seen)
    
    if not best:
        best = [0]
    
    return best
