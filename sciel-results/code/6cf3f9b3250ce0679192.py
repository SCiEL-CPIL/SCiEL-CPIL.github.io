"""Find a long simple path in the pinned G(100, 1.8/100) graph."""
import time
import sys
sys.setrecursionlimit(100000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    # Sort neighbors by degree (ascending) for better heuristic
    for i in range(n):
        adj[i] = sorted(adj[i], key=lambda x: (len(adj[x]), x))
    return adj

def _greedy_path(adj, start):
    """Greedy path extension from start vertex."""
    path = [start]
    seen = {start}
    while True:
        current = path[-1]
        candidates = [v for v in adj[current] if v not in seen]
        if not candidates:
            break
        # Pick the neighbor with minimum degree (Warnsdorff-like heuristic)
        v = min(candidates, key=lambda w: (len(adj[w]), w))
        path.append(v)
        seen.add(v)
    return path

def _backtrack_search(adj, best_path, deadline, node_limit):
    """
    Depth-limited DFS with pruning to find a longer path.
    Tries to extend from the ends of the current best path.
    """
    n = P['n']
    best = list(best_path)
    best_len = len(best)
    nodes_visited = [0]
    
    def dfs(path, seen, remaining_time_check):
        if nodes_visited[0] > node_limit:
            return
        if time.time() > deadline:
            return
        nodes_visited[0] += 1
        
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best = list(path)
        
        # If we've found a Hamiltonian path, we're done
        if current_len == n:
            return
        
        current = path[-1]
        # Try extending from the end
        for v in adj[current]:
            if v not in seen:
                path.append(v)
                seen.add(v)
                dfs(path, seen, remaining_time_check)
                path.pop()
                seen.remove(v)
                if time.time() > deadline or nodes_visited[0] > node_limit:
                    return
        
        # Also try extending from the beginning (reverse direction)
        if len(path) > 0:
            first = path[0]
            for v in adj[first]:
                if v not in seen:
                    path.insert(0, v)
                    seen.add(v)
                    dfs(path, seen, remaining_time_check)
                    path.pop(0)
                    seen.remove(v)
                    if time.time() > deadline or nodes_visited[0] > node_limit:
                        return
    
    # Start DFS from the current best path
    seen = set(best)
    dfs(list(best), seen, None)
    
    return best

def _improve_path(adj, path, deadline, max_extensions=50000):
    """
    Try to extend the path from both ends and also try re-routing.
    Uses iterative deepening with time limit.
    """
    n = P['n']
    best = list(path)
    best_len = len(best)
    
    # Try extending from both ends
    for attempt in range(100):
        if time.time() > deadline:
            break
        extended = False
        
        # Try extending from the end
        current = best[-1]
        for v in adj[current]:
            if v not in best:
                new_path = best + [v]
                # Try to extend further from new end
                while True:
                    last = new_path[-1]
                    cands = [w for w in adj[last] if w not in new_path]
                    if not cands:
                        break
                    # Pick min degree
                    w = min(cands, key=lambda x: (len(adj[x]), x))
                    new_path.append(w)
                if len(new_path) > best_len:
                    best = new_path
                    best_len = len(best)
                    extended = True
                    break
        
        if extended:
            continue
            
        # Try extending from the beginning
        current = best[0]
        for v in adj[current]:
            if v not in best:
                new_path = [v] + best
                while True:
                    first = new_path[0]
                    cands = [w for w in adj[first] if w not in new_path]
                    if not cands:
                        break
                    w = min(cands, key=lambda x: (len(adj[x]), x))
                    new_path.insert(0, w)
                if len(new_path) > best_len:
                    best = new_path
                    best_len = len(best)
                    extended = True
                    break
        
        if not extended:
            break
    
    return best

def _try_all_starts(adj, deadline):
    """Try greedy paths from all starts, keep the best."""
    best = []
    for start in range(P['n']):
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start)
        if len(path) > len(best):
            best = path
    return best

def _dfs_from_vertex(adj, start, target_len, deadline, node_limit):
    """
    DFS trying to find a path of at least target_len starting from start.
    Uses Warnsdorff heuristic for ordering.
    """
    n = P['n']
    found = [None]
    nodes = [0]
    
    def dfs(path, seen):
        if found[0] is not None:
            return
        if nodes[0] > node_limit or time.time() > deadline:
            return
        nodes[0] += 1
        
        if len(path) >= target_len:
            found[0] = list(path)
            return
        
        current = path[-1]
        # Order neighbors: prefer low degree, then unvisited
        cands = [v for v in adj[current] if v not in seen]
        cands.sort(key=lambda x: (len(adj[x]), x))
        
        for v in cands:
            if found[0] is not None:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
    
    dfs([start], {start})
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    adj = _build_adjacency()
    n = P['n']
    
    # Phase 1: Greedy from all starts
    best = _try_all_starts(adj, deadline)
    best_len = len(best)
    
    # Phase 2: Try to improve by extending from ends
    if time.time() < deadline - 5.0:
        best = _improve_path(adj, best, deadline)
        best_len = len(best)
    
    # Phase 3: DFS from promising starts targeting longer paths
    if time.time() < deadline - 10.0:
        # Try DFS from each vertex aiming for best_len + 1
        for start in range(n):
            if time.time() > deadline - 10.0:
                break
            if len(adj[start]) < 2:
                continue
            # Node limit per DFS call
            remaining = deadline - 10.0 - time.time()
            node_limit = max(10000, int(remaining * 50000))
            path = _dfs_from_vertex(adj, start, best_len + 1, deadline - 10.0, node_limit)
            if path is not None:
                if len(path) > best_len:
                    best = path
                    best_len = len(best)
                    # Try to extend further
                    if time.time() < deadline - 5.0:
                        best = _improve_path(adj, best, deadline)
                        best_len = len(best)
    
    # Phase 4: Final backtracking search from the best path
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        node_limit = max(50000, int(remaining * 100000))
        best = _backtrack_search(adj, best, deadline - 5.0, node_limit)
    
    # Ensure validity
    if not best:
        return [0]
    
    # Validate
    for i in range(len(best) - 1):
        if best[i+1] not in adj[best[i]]:
            # Fallback to a valid single vertex or edge
            return [best[0]]
    
    return best

# Pre-compute at module level for speed
_adj = _build_adjacency()
_initial_best = _try_all_starts(_adj, time.time() + 5.0)
_initial_len = len(_initial_best)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    adj = _adj
    n = P['n']
    
    # Start from pre-computed best
    best = list(_initial_best)
    best_len = _initial_len
    
    # Phase 1: Try to improve by extending from ends
    if time.time() < deadline - 5.0:
        best = _improve_path(adj, best, deadline)
        best_len = len(best)
    
    # Phase 2: DFS from promising starts targeting longer paths
    if time.time() < deadline - 10.0:
        # Try DFS from each vertex aiming for best_len + 1
        # Prioritize starts with higher degree
        starts = sorted(range(n), key=lambda x: -len(adj[x]))
        for start in starts:
            if time.time() > deadline - 10.0:
                break
            if len(adj[start]) < 2:
                continue
            remaining = deadline - 10.0 - time.time()
            if remaining < 0.1:
                break
            node_limit = max(10000, int(remaining * 50000))
            path = _dfs_from_vertex(adj, start, best_len + 1, deadline - 10.0, node_limit)
            if path is not None:
                if len(path) > best_len:
                    best = path
                    best_len = len(best)
                    # Try to extend further
                    if time.time() < deadline - 5.0:
                        best = _improve_path(adj, best, deadline)
                        best_len = len(best)
    
    # Phase 3: Final backtracking search from the best path
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        if remaining > 0.1:
            node_limit = max(50000, int(remaining * 100000))
            best = _backtrack_search(adj, best, deadline - 5.0, node_limit)
    
    # Ensure validity
    if not best:
        return [0]
    
    # Validate
    for i in range(len(best) - 1):
        if best[i+1] not in adj[best[i]]:
            return [best[0]]
    
    return best
