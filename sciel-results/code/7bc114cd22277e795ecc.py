import time
import sys
from itertools import combinations
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    # Precompute common neighbors for all pairs
    common_neighbors = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if u not in adj_set[v]:
                cn = adj_set[u] & adj_set[v]
                common_neighbors[u][v] = cn
                common_neighbors[v][u] = cn
    
    def time_check():
        return time.time() > deadline
    
    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    def check_feasible(branches):
        """Check if branches form a valid set. Returns assignment or None."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        non_direct_pairs = []
        for u, v in pairs:
            if u not in adj_set[v]:
                cn = common_neighbors[u][v] - branch_set
                if not cn:
                    return None
                non_direct_pairs.append((u, v, cn))
        
        if not non_direct_pairs:
            return {p: [] for p in pairs}
        
        # Try to assign distinct internal vertices
        # Sort by fewest options
        non_direct_pairs.sort(key=lambda x: len(x[2]))
        
        assignment = {}
        used = set()
        
        def bt(idx):
            if time_check():
                return False
            if idx == len(non_direct_pairs):
                return True
            u, v, candidates = non_direct_pairs[idx]
            for w in candidates:
                if w not in used:
                    used.add(w)
                    assignment[(u, v)] = [w]
                    if bt(idx + 1):
                        return True
                    used.remove(w)
                    del assignment[(u, v)]
            return False
        
        if not bt(0):
            # Try BFS for longer paths
            used_internals = set()
            assignment2 = {}
            
            def bt_bfs(idx):
                if time_check():
                    return False
                if idx == len(non_direct_pairs):
                    return True
                u, v, _ = non_direct_pairs[idx]
                blocked = branch_set | used_internals
                blocked.discard(u)
                blocked.discard(v)
                path = bfs_path(u, v, blocked)
                if path is None:
                    return False
                internal = path[1:-1]
                if len(internal) != len(set(internal)):
                    return False
                for w in internal:
                    if w in branch_set:
                        return False
                for w in internal:
                    used_internals.add(w)
                assignment2[(u, v)] = internal
                if bt_bfs(idx + 1):
                    return True
                for w in internal:
                    used_internals.discard(w)
                del assignment2[(u, v)]
                return False
            
            if not bt_bfs(0):
                return None
            return assignment2
        
        # Build full result
        result = {}
        for u, v in pairs:
            if u in adj_set[v]:
                result[(u, v)] = []
            else:
                result[(u, v)] = assignment[(u, v)]
        return result
    
    def solve():
        best_result = None
        best_k = 0
        
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        # Greedy growth from multiple seeds
        for seed in sorted_vertices[:10]:
            if time_check():
                break
            current = [seed]
            for v in sorted_vertices:
                if v == seed:
                    continue
                current.append(v)
                if check_feasible(current) is None:
                    current.pop()
                if time_check():
                    break
            if len(current) > best_k:
                assignment = check_feasible(current)
                if assignment is not None:
                    paths = {}
                    for (u, v), internal in assignment.items():
                        paths[(u, v)] = [u] + internal + [v]
                    best_result = {
                        'branches': [int(b) for b in current],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current)
        
        # Backtracking to improve
        def backtrack(current_set, start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            if len(current_set) > best_k:
                assignment = check_feasible(current_set)
                if assignment is not None:
                    paths = {}
                    for (u, v), internal in assignment.items():
                        paths[(u, v)] = [u] + internal + [v]
                    best_result = {
                        'branches': [int(b) for b in current_set],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_set)
            
            for i in range(start_idx, n):
                if time_check():
                    return
                candidate = sorted_vertices[i]
                if candidate in current_set:
                    continue
                new_set = current_set + [candidate]
                if check_feasible(new_set) is not None:
                    backtrack(new_set, i + 1)
        
        # Start backtracking from small sets
        for init_size in range(2, min(4, best_k + 1)):
            if time_check():
                break
            for combo in combinations(sorted_vertices[:15], init_size):
                if time_check():
                    break
                if check_feasible(list(combo)) is not None:
                    backtrack(list(combo), 0)
        
        if best_result is not None:
            return best_result
        
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
# EVOLVE-BLOCK-END
