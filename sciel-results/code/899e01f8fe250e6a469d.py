"""Longest path in sparse G(100, 0.018) via backtracking with heuristics."""
import time
import sys
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Precompute degree for heuristic ordering
degrees = [len(adj[i]) for i in range(N)]

def backtrack_path(start, max_depth=None, time_limit=None, t0=None):
    """Find longest path starting from 'start' using backtracking with Warnsdorff heuristic."""
    if time_limit is None:
        time_limit = 50.0
    if t0 is None:
        t0 = time.time()
    
    best_path = [start]
    best_len = 1
    
    # Use iterative deepening: try to extend path greedily first, then backtrack
    path = [start]
    visited = {start}
    
    def dfs(current, path, visited):
        nonlocal best_path, best_len
        if time.time() - t0 > time_limit:
            return False  # timeout signal
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        neighbors = [v for v in adj[current] if v not in visited]
        if not neighbors:
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        neighbors.sort(key=lambda v: (len(adj[v] - visited), v))
        
        for v in neighbors:
            path.append(v)
            visited.add(v)
            if max_depth is not None and len(path) >= max_depth:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                path.pop()
                visited.remove(v)
                continue
            dfs(v, path, visited)
            if time.time() - t0 > time_limit:
                path.pop()
                visited.remove(v)
                return True  # timeout
            path.pop()
            visited.remove(v)
        
        return False
    
    dfs(start, path, visited)
    return best_path

def greedy_extend(start):
    """Quick greedy extension to get a baseline path."""
    path = [start]
    visited = {start}
    while True:
        neighbors = [v for v in adj[path[-1]] if v not in visited]
        if not neighbors:
            break
        # Pick neighbor with fewest unvisited neighbors (Warnsdorff)
        v = min(neighbors, key=lambda w: (len(adj[w] - visited), w))
        path.append(v)
        visited.add(v)
    return path

def construct():
    t0 = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    best_path = []
    
    # First pass: quick greedy from all vertices to get initial best
    for start in range(N):
        if time.time() - t0 > deadline:
            break
        p = greedy_extend(start)
        if len(p) > len(best_path):
            best_path = p
    
    # Sort starts by greedy path length descending - prioritize promising starts
    start_order = sorted(range(N), key=lambda s: -len(greedy_extend(s)))
    
    # Second pass: systematic backtracking from most promising starts
    for start in start_order:
        if time.time() - t0 > deadline:
            break
        p = backtrack_path(start, time_limit=deadline - (time.time() - t0), t0=t0)
        if len(p) > len(best_path):
            best_path = p
    
    # Third pass: try remaining starts if time permits
    if time.time() - t0 < deadline - 5:
        for start in range(N):
            if time.time() - t0 > deadline:
                break
            if start in start_order[:20]:  # Already tried top 20
                continue
            p = backtrack_path(start, time_limit=deadline - (time.time() - t0), t0=t0)
            if len(p) > len(best_path):
                best_path = p
    
    return best_path if best_path else [0]
