"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Precompute adjacency list
_N = P['n']
_ADJ = [set() for _ in range(_N)]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

# Convert to lists for faster iteration
_ADJ_LIST = [list(s) for s in _ADJ]

def _greedy_extend(path, seen):
    """Extend path greedily from both ends using Warnsdorff's rule."""
    # Extend from end
    while True:
        candidates = [v for v in _ADJ[path[-1]] if v not in seen]
        if not candidates:
            break
        # Warnsdorff: choose candidate with fewest unvisited neighbors
        best_v = min(candidates, key=lambda v: (len([u for u in _ADJ[v] if u not in seen]), v))
        path.append(best_v)
        seen.add(best_v)
    
    # Extend from start
    while True:
        candidates = [v for v in _ADJ[path[0]] if v not in seen]
        if not candidates:
            break
        best_v = min(candidates, key=lambda v: (len([u for u in _ADJ[v] if u not in seen]), v))
        path.insert(0, best_v)
        seen.add(best_v)
    
    return path, seen

def _dfs_warnsdorff(start, max_depth, deadline):
    """
    DFS with Warnsdorff heuristic and backtracking to find longest path.
    Returns the best path found.
    """
    best_path = [start]
    best_len = 1
    
    path = [start]
    visited = [False] * _N
    visited[start] = True
    
    def dfs(current_depth):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return False
        
        last = path[-1]
        candidates = [v for v in _ADJ[last] if not visited[v]]
        
        if not candidates:
            # Dead end: update best if this path is longer
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (ascending)
        candidates.sort(key=lambda v: (sum(1 for u in _ADJ[v] if not visited[u]), v))
        
        for v in candidates:
            if time.time() > deadline:
                return False
            
            path.append(v)
            visited[v] = True
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            # If we've reached max_depth, stop
            if len(path) >= max_depth:
                path.pop()
                visited[v] = False
                return True
            
            # Recurse
            dfs(current_depth + 1)
            
            # Backtrack
            path.pop()
            visited[v] = False
            
            # If we found a path of max_depth, we can stop
            if best_len >= max_depth:
                return True
        
        return False
    
    dfs(0)
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Phase 1: Greedy baseline from each vertex
    best_path = []
    best_len = 0
    
    for start in range(_N):
        if time.time() > deadline:
            break
        
        path = [start]
        seen = {start}
        path, seen = _greedy_extend(path, seen)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
            if best_len == _N:
                return best_path
    
    # Phase 2: DFS with Warnsdorff and backtracking from promising starts
    # Try from vertices not in the best path first, then from all vertices
    in_best = set(best_path)
    starts = [v for v in range(_N) if v not in in_best] + [v for v in range(_N) if v in in_best]
    
    for start in starts:
        if time.time() > deadline:
            break
        
        # Give each start a reasonable time budget
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Allocate time: more time for earlier starts
        time_budget = min(remaining * 0.3, 5.0)
        local_deadline = time.time() + time_budget
        
        # Try DFS with increasing max_depth
        for max_depth in range(best_len + 1, _N + 1, 5):
            if time.time() > local_deadline:
                break
            
            path = _dfs_warnsdorff(start, max_depth, local_deadline)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
                if best_len == _N:
                    return best_path
    
    # Phase 3: If we have time, do more targeted searches with smaller time budgets
    if time.time() < deadline and best_len < _N:
        for start in range(_N):
            if time.time() > deadline:
                break
            
            remaining = deadline - time.time()
            if remaining < 0.5:
                break
            
            time_budget = min(remaining, 2.0)
            local_deadline = time.time() + time_budget
            
            # Try full DFS with current best length + 1
            path = _dfs_warnsdorff(start, best_len + 1, local_deadline)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
                if best_len == _N:
                    return best_path
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
