"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_path(adj, start, strategy='min_remaining'):
    """Greedy path construction with different strategies."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'min_remaining':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_remaining':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
        else:
            v = min(candidates)
        path.append(v)
        seen.add(v)
    return path

def _warnsdorff_path(adj, start, max_nodes=50000):
    """
    Depth-first search with Warnsdorff heuristic (prefer neighbors with fewest unvisited neighbors).
    This is often more effective than simple greedy for finding long paths in sparse graphs.
    """
    n = P['n']
    best_path = [start]
    current_path = [start]
    seen = {start}
    
    # Pre-compute initial degrees for heuristic
    # We'll compute "remaining degree" on the fly to be accurate
    
    def dfs(node, path, seen_set):
        nonlocal best_path, max_nodes, current_path
        if len(path) > len(best_path):
            best_path = list(path)
            current_path = list(path)
        
        if len(path) == n:
            return True
            
        if len(path) > max_nodes:
            return False
            
        neighbors = sorted(adj[node] - seen_set, key=lambda w: len(adj[w] - seen_set))
        
        for next_node in neighbors:
            path.append(next_node)
            seen_set.add(next_node)
            if dfs(next_node, path, seen_set):
                return True
            path.pop()
            seen_set.remove(next_node)
            
        return False

    # Run DFS
    dfs(start, current_path, seen)
    
    return best_path

def _try_extend_path(path, adj, max_extensions=50):
    """Try to extend a path from either end."""
    if not path:
        return path
    path = list(path)
    seen = set(path)
    extended = True
    count = 0
    while extended and count < max_extensions:
        extended = False
        count += 1
        # Try extending from the end
        if path:
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                extended = True
        # Try extending from the beginning
        if path:
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                extended = True
    return path

def _two_opt_improve(path, adj, max_iter=200):
    """Improve path length using insertion and extension."""
    if len(path) < 3:
        return path
    path = list(path)
    seen = set(path)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        # Try to find a vertex not in path that can be inserted
        for i in range(len(path) - 1):
            u, w = path[i], path[i + 1]
            common = (adj[u] & adj[w]) - seen
            if common:
                v = min(common)
                path = path[:i+1] + [v] + path[i+1:]
                seen.add(v)
                improved = True
                break
        # Also try to extend ends
        if not improved and path:
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                improved = True
        if not improved and path:
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                improved = True
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    n = P['n']
    adj = _build_adj()
    
    best_path = []
    
    # Strategy 1: Greedy from every start vertex (fast)
    for start in range(n):
        if time.time() > deadline:
            break
        for strat in ['min_remaining', 'max_remaining']:
            path = _greedy_path(adj, start, strategy=strat)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 2: Warnsdorff DFS from promising starts (low degree vertices often lead to longer paths)
    # Sort vertices by degree to try constrained ones first
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x])
    
    # Try Warnsdorff on a subset of starts to avoid explosion
    for start in sorted_vertices[:20]:
        if time.time() > deadline:
            break
        # Limit nodes to prevent timeout on hard instances
        path = _warnsdorff_path(adj, start, max_nodes=10000)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 3: Improve best path with local search
    if len(best_path) > 0:
        improved = _two_opt_improve(best_path, adj, max_iter=100)
        if len(improved) > len(best_path):
            best_path = improved
        
        # Try extending ends
        extended = _try_extend_path(best_path, adj, max_extensions=100)
        if len(extended) > len(best_path):
            best_path = extended
    
    # Strategy 4: If we still have time, try Warnsdorff from other starts or more nodes
    if time.time() < deadline - 5:
        for start in sorted_vertices[20:50]:
            if time.time() > deadline:
                break
            path = _warnsdorff_path(adj, start, max_nodes=5000)
            if len(path) > len(best_path):
                best_path = path
                # Improve immediately
                improved = _two_opt_improve(best_path, adj, max_iter=50)
                if len(improved) > len(best_path):
                    best_path = improved

    # Final polish
    if len(best_path) > 0 and time.time() < deadline:
        final = _try_extend_path(best_path, adj, max_extensions=100)
        final = _two_opt_improve(final, adj, max_iter=100)
        if len(final) > len(best_path):
            best_path = final
    
    # Ensure we return a valid nonempty path
    if not best_path:
        for u, v in P['edges']:
            best_path = [u, v]
            break
        if not best_path:
            best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
