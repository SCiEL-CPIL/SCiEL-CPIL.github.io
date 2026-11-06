import time
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline

    # Strategy 1: Greedy Warnsdorff to get a baseline
    for start in range(n):
        if time_check():
            break
        path = [start]
        visited = {start}
        while True:
            curr = path[-1]
            candidates = [v for v in adj[curr] if v not in visited]
            if not candidates:
                break
            # Sort by number of unvisited neighbors (Warnsdorff)
            candidates.sort(key=lambda v: len(adj[v] - visited))
            next_v = candidates[0]
            path.append(next_v)
            visited.add(next_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    if len(best_path) == n:
        return best_path

    # Strategy 2: DFS with Warnsdorff heuristic
    # We try to find a path longer than best_path
    # We use recursion with backtracking
    
    visited = [False] * n
    path = []
    
    def dfs(u):
        nonlocal best_path, time_check, deadline
        if time_check():
            return False
        
        path.append(u)
        visited[u] = True
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        if len(path) == n:
            return True
        
        curr = u
        candidates = [v for v in adj[curr] if not visited[v]]
        
        if not candidates:
            path.pop()
            visited[u] = False
            return False
        
        # Sort candidates: prefer those with fewer unvisited neighbors (Warnsdorff)
        # This helps avoid creating isolated vertices
        candidates.sort(key=lambda v: len(adj[v] - {x for x in range(n) if visited[x]}))
        
        # Optional: Check for forced moves
        # If a candidate v has only 1 unvisited neighbor (curr), we MUST go there
        # But since we are sorting by remaining degree, v with degree 1 will be first anyway.
        
        for v in candidates:
            if time_check():
                break
            if dfs(v):
                # If we found a full path, we can unwind
                # But we need to clean up before returning True to caller?
                # Actually, if we return True, the caller also returns True, 
                # and we should not pop in the success case if we want to keep the path.
                # But the standard DFS pattern pops before return.
                # Let's handle cleanup carefully.
                pass
            
        # If we didn't return True from the loop, we failed to extend
        path.pop()
        visited[u] = False
        return False

    # We can start DFS from different vertices to increase chances
    # But DFS is expensive. Let's try starting from a few low-degree vertices or just 0.
    # Actually, the greedy pass already explored all starts.
    # Let's just run DFS from the start vertex of the current best_path, 
    # or a few others.
    
    # To be safe, let's run DFS from vertex 0, and if that fails to improve, try 1, 2...
    # But time is tight.
    
    # Let's try a single deep DFS from the best start found by greedy?
    # Or just start from 0.
    
    # Actually, a better approach for sparse graphs is to use the greedy result as a seed
    # and try to extend or rearrange. But DFS from scratch is more robust.
    
    # Let's run DFS from 0
    dfs(0)
    
    if len(best_path) == n:
        return best_path

    # If DFS from 0 didn't find a full path, try a few other starts
    for start in range(1, 10):
        if time_check():
            break
        if len(best_path) == n:
            break
        dfs(start)
        
    return best_path if best_path else [0]
