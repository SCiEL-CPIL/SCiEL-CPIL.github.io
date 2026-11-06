"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 80  # safety margin
    
    def greedy_path(start, heuristic):
        """Greedy path from start using given heuristic."""
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                break
            if heuristic == 'min_deg':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif heuristic == 'max_deg':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif heuristic == 'min_deg_global':
                v = min(candidates, key=lambda w: (len(adj[w]), w))
            elif heuristic == 'max_deg_global':
                v = max(candidates, key=lambda w: (len(adj[w]), w))
            else:
                v = min(candidates)
            path.append(v)
            seen.add(v)
        return path, seen
    
    def try_extend(path, seen):
        """Try to extend path from both ends."""
        extended = True
        while extended:
            extended = False
            # Try extending from the end
            cur = path[-1]
            for v in sorted(adj[cur] - seen):
                path.append(v)
                seen.add(v)
                extended = True
                break
            # Try extending from the front
            cur = path[0]
            for v in sorted(adj[cur] - seen):
                path.insert(0, v)
                seen.add(v)
                extended = True
                break
        return path, seen
    
    def try_insert(path, seen):
        """Try to insert unvisited vertices into the path."""
        improved = True
        while improved:
            improved = False
            unvisited = set(range(n)) - seen
            if not unvisited:
                break
            # Try inserting each unvisited vertex at each position
            for v in sorted(unvisited):
                for i in range(len(path)):
                    # Can we insert v between path[i] and path[i+1]?
                    if i < len(path) - 1:
                        a, b = path[i], path[i+1]
                        if v in adj[a] and v in adj[b]:
                            path.insert(i + 1, v)
                            seen.add(v)
                            improved = True
                            break
                    # Can we insert v at the end?
                    if not improved and i == len(path) - 1:
                        if v in adj[path[-1]]:
                            path.append(v)
                            seen.add(v)
                            improved = True
                            break
                    # Can we insert v at the front?
                    if not improved and i == 0:
                        if v in adj[path[0]]:
                            path.insert(0, v)
                            seen.add(v)
                            improved = True
                            break
                if improved:
                    break
        return path, seen
    
    def backtrack_extend(path, seen, max_depth=50):
        """Try to extend using limited backtracking."""
        best_path = list(path)
        best_seen = set(seen)
        
        def dfs(cur_path, cur_seen, depth):
            nonlocal best_path, best_seen
            if time.time() > deadline:
                return
            if len(cur_path) > len(best_path):
                best_path = list(cur_path)
                best_seen = set(cur_seen)
            if depth >= max_depth:
                return
            cur = cur_path[-1]
            for v in sorted(adj[cur] - cur_seen):
                cur_path.append(v)
                cur_seen.add(v)
                dfs(cur_path, cur_seen, depth + 1)
                cur_path.pop()
                cur_seen.remove(v)
        
        dfs(path, seen, 0)
        return best_path, best_seen
    
    best_path = []
    best_seen = set()
    
    # Phase 1: Multiple greedy strategies from all starts
    for start in range(n):
        if time.time() > deadline:
            break
        for h in ['min_deg', 'max_deg', 'min_deg_global', 'max_deg_global']:
            if time.time() > deadline:
                break
            path, seen = greedy_path(start, h)
            path, seen = try_extend(path, seen)
            if len(path) > len(best_path):
                best_path = path
                best_seen = seen
    
    # Phase 2: Local improvement on best path
    if best_path:
        best_path, best_seen = try_insert(best_path, best_seen)
    
    # Phase 3: Try backtracking extension from best path
    if best_path and time.time() < deadline - 10:
        extended_path, extended_seen = backtrack_extend(best_path, best_seen, max_depth=30)
        if len(extended_path) > len(best_path):
            best_path = extended_path
            best_seen = extended_seen
            best_path, best_seen = try_insert(best_path, best_seen)
    
    # Phase 4: Try more greedy with different starts, using improved best as reference
    for start in range(n):
        if time.time() > deadline - 5:
            break
        if start in best_seen:
            continue
        for h in ['min_deg', 'max_deg']:
            if time.time() > deadline - 5:
                break
            path, seen = greedy_path(start, h)
            path, seen = try_extend(path, seen)
            path, seen = try_insert(path, seen)
            if len(path) > len(best_path):
                best_path = path
                best_seen = seen
    
    # Final: ensure non-empty
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
