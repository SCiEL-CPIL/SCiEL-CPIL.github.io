"""Long path finder for pinned G(100, 1.8/100) graph using Warnsdorff-style greedy with backtracking."""

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _warnsdorff_path(start, adj, n):
    """Greedy path using Warnsdorff's rule: always move to neighbor with fewest unvisited neighbors."""
    path = [start]
    seen = {start}
    while True:
        curr = path[-1]
        candidates = [v for v in adj[curr] if v not in seen]
        if not candidates:
            break
        # Warnsdorff: pick candidate with minimum number of unvisited neighbors
        def key(v):
            return (len(adj[v] - seen), v)
        v = min(candidates, key=key)
        path.append(v)
        seen.add(v)
    return path

def _try_extend(path, adj, n):
    """Try to extend path at both ends by one step if possible (greedy)."""
    changed = True
    while changed:
        changed = False
        # Try extending at end
        if len(path) < n:
            end = path[-1]
            cands = [v for v in adj[end] if v not in set(path)]
            if cands:
                v = min(cands, key=lambda x: (len(adj[x] - set(path)), x))
                path.append(v)
                changed = True
        # Try extending at start
        if len(path) < n:
            start = path[0]
            cands = [v for v in adj[start] if v not in set(path)]
            if cands:
                v = min(cands, key=lambda x: (len(adj[x] - set(path)), x))
                path.insert(0, v)
                changed = True
    return path

def _limited_backtracking(start, adj, n, max_depth=15, time_budget=2.0):
    """Backtracking search with Warnsdorff ordering to find longer paths.
       Limits recursion depth to avoid explosion."""
    import time
    start_time = time.time()
    best = [start]
    
    def dfs(path, seen):
        nonlocal best
        if time.time() - start_time > time_budget:
            return
        if len(path) > len(best):
            best = path[:]
        curr = path[-1]
        cands = [v for v in adj[curr] if v not in seen]
        if not cands:
            return
        # Sort by Warnsdorff heuristic
        cands.sort(key=lambda v: (len(adj[v] - seen), v))
        for v in cands:
            if len(path) >= n:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() - start_time > time_budget:
                return
    
    dfs([start], {start})
    return best

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    best_path = []
    
    # Phase 1: Try Warnsdorff greedy from every node
    for start in range(n):
        path = _warnsdorff_path(start, adj, n)
        path = _try_extend(path, adj, n)
        if len(path) > len(best_path):
            best_path = path
    
    # Phase 2: Try limited backtracking from promising starts (low-degree nodes first)
    degree = [len(adj[i]) for i in range(n)]
    # Sort nodes by degree ascending (low-degree nodes are better starts for Warnsdorff)
    starts = sorted(range(n), key=lambda i: (degree[i], i))
    
    # Try backtracking from a few of the best starts
    for start in starts[:10]:
        path = _limited_backtracking(start, adj, n, max_depth=12, time_budget=1.5)
        path = _try_extend(path, adj, n)
        if len(path) > len(best_path):
            best_path = path
    
    # Phase 3: Try to improve best_path by local insertion of unvisited nodes
    # This is a simple local search: for each unvisited node, try to insert it
    unvisited = set(range(n)) - set(best_path)
    improved = True
    while improved:
        improved = False
        for u in list(unvisited):
            # Try inserting u between consecutive pairs in best_path
            inserted = False
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i + 1]
                if u in adj[a] and u in adj[b]:
                    # Check that u is not already in path
                    if u not in best_path:
                        new_path = best_path[:i+1] + [u] + best_path[i+1:]
                        # Validate
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in adj[new_path[j]]:
                                valid = False
                                break
                        if valid:
                            best_path = new_path
                            unvisited.discard(u)
                            improved = True
                            inserted = True
                            break
            if inserted:
                break
            # Also try prepending or appending
            if not inserted and u in adj[best_path[0]] and u not in best_path:
                best_path = [u] + best_path
                unvisited.discard(u)
                improved = True
            elif not inserted and u in adj[best_path[-1]] and u not in best_path:
                best_path = best_path + [u]
                unvisited.discard(u)
                improved = True
    
    return best_path
