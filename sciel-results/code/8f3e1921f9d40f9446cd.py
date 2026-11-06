"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: (degree[x], x))
    
    best_path = []
    best_len = 0
    deadline = time.time() + 85  # Leave 5s margin before 90s kill
    
    def try_extend_both_ends(path, visited, time_limit):
        """Try to extend the path from both ends using greedy + limited backtracking."""
        nonlocal best_path, best_len
        path = list(path)
        visited = set(visited)
        
        def extend_end(end_idx, time_limit):
            """Extend from one end. end_idx=0 for left, -1 for right."""
            nonlocal best_path, best_len
            while True:
                if time.time() > time_limit:
                    return
                current = path[0] if end_idx == 0 else path[-1]
                unvisited_neighbors = [v for v in adj[current] if v not in visited]
                if not unvisited_neighbors:
                    return
                # Warnsdorff: prefer neighbors with fewer unvisited options
                unvisited_neighbors.sort(key=lambda v: (len(adj[v] - visited), v))
                # Try up to 3 options with limited backtracking
                found = False
                for v in unvisited_neighbors[:3]:
                    if time.time() > time_limit:
                        return
                    if end_idx == 0:
                        path.insert(0, v)
                    else:
                        path.append(v)
                    visited.add(v)
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                    # Try to extend further from this new end
                    old_end = end_idx
                    if end_idx == 0:
                        end_idx = 0  # still left
                    else:
                        end_idx = -1  # still right
                    # Recursively try to extend more
                    old_path_len = len(path)
                    extend_end(end_idx, time_limit)
                    # Check if we extended
                    if len(path) > old_path_len:
                        found = True
                        break
                    else:
                        # Backtrack
                        if end_idx == 0:
                            path.pop(0)
                        else:
                            path.pop()
                        visited.remove(v)
                if not found:
                    return
        
        # Alternate extending from both ends
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            old_len = len(path)
            extend_end(0, time_limit)
            if len(path) > old_len:
                changed = True
                old_len = len(path)
            extend_end(-1, time_limit)
            if len(path) > old_len:
                changed = True
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    def dfs_warnsdorff(start, time_limit):
        """DFS with Warnsdorff heuristic and connectivity pruning."""
        nonlocal best_path, best_len
        path = [start]
        visited = {start}
        
        # Precompute adjacency as lists for speed
        adj_lists = [sorted(adj[i]) for i in range(n)]
        
        def _dfs():
            nonlocal best_path, best_len
            if time.time() > time_limit:
                return False
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            current = path[-1]
            unvisited_neighbors = [v for v in adj[current] if v not in visited]
            if not unvisited_neighbors:
                return True
            
            # Connectivity check: ensure remaining unvisited vertices that are
            # adjacent to visited vertices can still be reached
            # (lightweight pruning: skip if adding v would disconnect some 
            #  unvisited vertex from the rest)
            # For speed, only do full check at deeper levels
            if len(path) > 10:
                # Check if any unvisited vertex becomes isolated
                for v in unvisited_neighbors:
                    # After visiting v, check if any unvisited neighbor of v
                    # has no other unvisited neighbors and isn't the current end
                    pass  # Skip expensive check for now
            
            # Sort by Warnsdorff heuristic
            unvisited_neighbors.sort(key=lambda v: (len(adj[v] - visited), v))
            
            for v in unvisited_neighbors:
                if time.time() > time_limit:
                    return True
                path.append(v)
                visited.add(v)
                if not _dfs():
                    return True
                path.pop()
                visited.remove(v)
            return True
        
        _dfs()
        
        # After DFS, try to extend the best path found from both ends
        if best_path and len(best_path) >= 2:
            try_extend_both_ends(best_path, set(best_path), time_limit)
        
        return best_path
    
    # Phase 1: Try starting from each vertex, prioritizing low-degree vertices
    for start in vertices_by_degree:
        if time.time() > deadline:
            break
        if degree[start] == 0:
            continue
        time_per_start = 0.4
        dfs_warnsdorff(start, time.time() + time_per_start)
    
    # Phase 2: Two-end extension on the best path found
    if best_path and time.time() < deadline - 2:
        try_extend_both_ends(best_path, set(best_path), time.time() + 2)
    
    # Phase 3: Randomized restarts with two-end extension
    if time.time() < deadline - 1:
        rng = random.Random(42)
        attempts = 0
        while time.time() < deadline - 1 and attempts < 30:
            start = rng.randint(0, n - 1)
            if degree[start] == 0:
                attempts += 1
                continue
            time_per_attempt = 0.3
            dfs_warnsdorff(start, time.time() + time_per_attempt)
            if best_path and len(best_path) >= 2:
                try_extend_both_ends(best_path, set(best_path), time.time() + 0.5)
            attempts += 1
    
    # Phase 4: Final polish - try to extend the best path one more time
    if best_path and time.time() < deadline - 0.5:
        try_extend_both_ends(best_path, set(best_path), deadline - 0.5)
    
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
