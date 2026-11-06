"""Longest path in sparse G(100, 0.018) via two-ended backtracking with connectivity pruning."""
import time
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

adj_list = [list(s) for s in adj]
degrees = [len(adj[i]) for i in range(N)]

def check_connectivity(visited, path):
    """Check if unvisited vertices remain connected (necessary condition for extending path)."""
    unvisited = [v for v in range(N) if v not in visited]
    if len(unvisited) <= 1:
        return True
    # BFS from first unvisited vertex
    start = unvisited[0]
    seen = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited and v not in seen:
                seen.add(v)
                queue.append(v)
    return len(seen) == len(unvisited)

def greedy_two_end(start_left, start_right=None, max_iter=1000):
    """Grow path from both ends using Warnsdorff heuristic."""
    if start_right is None:
        start_right = start_left
    
    path = [start_left]
    visited = {start_left}
    right_end = start_right
    
    if right_end != start_left and right_end in adj[start_left]:
        path.append(right_end)
        visited.add(right_end)
    
    if len(path) == 1:
        # Single end growth
        while True:
            curr = path[-1]
            candidates = [v for v in adj[curr] if v not in visited]
            if not candidates:
                break
            candidates.sort(key=lambda v: (len(adj[v] - visited), v))
            path.append(candidates[0])
            visited.add(candidates[0])
        return path
    
    # Two-ended growth
    for _ in range(max_iter):
        left_candidates = [v for v in adj[path[0]] if v not in visited]
        right_candidates = [v for v in adj[path[-1]] if v not in visited]
        
        if not left_candidates and not right_candidates:
            break
        
        # Choose which end to extend
        if left_candidates and (not right_candidates or 
                                min(len(adj[v] - visited) for v in left_candidates) <= 
                                min(len(adj[v] - visited) for v in right_candidates)):
            left_candidates.sort(key=lambda v: (len(adj[v] - visited), v))
            v = left_candidates[0]
            path.insert(0, v)
        else:
            right_candidates.sort(key=lambda v: (len(adj[v] - visited), v))
            v = right_candidates[0]
            path.append(v)
        visited.add(v)
    
    return path

def backtrack_two_end(path, visited, best_path, deadline):
    """
    Backtracking search extending path from both ends.
    path: current path, visited: set of visited vertices
    Returns updated best_path
    """
    nonlocal_best = [best_path]
    
    def dfs():
        if time.time() > deadline:
            return False
        
        if len(path) > len(nonlocal_best[0]):
            nonlocal_best[0] = path[:]
        
        if len(path) == N:
            return True
        
        # Try extending from left end
        left_candidates = [v for v in adj[path[0]] if v not in visited]
        # Try extending from right end  
        right_candidates = [v for v in adj[path[-1]] if v not in visited]
        
        all_candidates = []
        for v in left_candidates:
            all_candidates.append((v, 'left'))
        for v in right_candidates:
            all_candidates.append((v, 'right'))
        
        if not all_candidates:
            return False
        
        # Sort by Warnsdorff: prefer vertices with fewer unvisited neighbors
        def sort_key(item):
            v, side = item
            unvisited_count = len(adj[v] - visited)
            # Also consider if this creates a forced move situation
            return (unvisited_count, v)
        
        all_candidates.sort(key=sort_key)
        
        for v, side in all_candidates:
            if time.time() > deadline:
                return False
            
            # Check connectivity before committing
            new_visited = visited | {v}
            
            # Quick connectivity check (only if we've made significant progress)
            if len(path) > 20 and not check_connectivity(new_visited, path):
                continue
            
            if side == 'left':
                path.insert(0, v)
            else:
                path.append(v)
            visited.add(v)
            
            if dfs():
                return True
            
            if side == 'left':
                path.pop(0)
            else:
                path.pop()
            visited.remove(v)
        
        return False
    
    dfs()
    return nonlocal_best[0]

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s limit (evaluator kills at 90s)
    
    best_path = []
    
    # Phase 1: Quick greedy from all vertices to get initial best
    for start in range(N):
        if time.time() > deadline:
            break
        p = greedy_two_end(start)
        if len(p) > len(best_path):
            best_path = p[:]
    
    if len(best_path) == N:
        return best_path
    
    # Phase 2: Sort starts by greedy length, prioritize promising ones
    start_scores = []
    for start in range(N):
        p = greedy_two_end(start)
        start_scores.append((len(p), start))
    start_scores.sort(key=lambda x: -x[0])
    
    # Phase 3: Systematic backtracking from most promising starts
    for _, start in start_scores:
        if time.time() > deadline:
            break
        
        # Initialize path with greedy result from this start
        greedy_p = greedy_two_end(start)
        if len(greedy_p) <= len(best_path):
            continue
        
        visited = set(greedy_p)
        path = greedy_p[:]
        
        # Try to extend via backtracking
        improved = backtrack_two_end(path, visited, best_path, deadline)
        if len(improved) > len(best_path):
            best_path = improved
        
        if len(best_path) == N:
            return best_path
    
    # Phase 4: Try different two-end combinations if time permits
    if time.time() < deadline - 5:
        # Try pairs of low-degree vertices as endpoints
        low_degree_starts = sorted(range(N), key=lambda v: degrees[v])
        tried_pairs = set()
        
        for i, s1 in enumerate(low_degree_starts[:30]):
            if time.time() > deadline:
                break
            for s2 in low_degree_starts[i+1:i+30]:
                if time.time() > deadline:
                    break
                if (s1, s2) in tried_pairs or (s2, s1) in tried_pairs:
                    continue
                if s1 not in adj[s2]:
                    # Need to find a path between them first
                    continue
                
                tried_pairs.add((s1, s2))
                
                # Build initial path between s1 and s2
                path = greedy_two_end(s1, s2)
                if len(path) <= len(best_path):
                    continue
                
                visited = set(path)
                improved = backtrack_two_end(path, visited, best_path, deadline)
                if len(improved) > len(best_path):
                    best_path = improved
                
                if len(best_path) == N:
                    return best_path
    
    return best_path if best_path else [0]
