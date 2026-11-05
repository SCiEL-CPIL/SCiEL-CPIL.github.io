import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    return components

def _bfs_longest_path_tree(adj, comp, comp_set):
    """For a tree component, find the diameter path (longest path)."""
    if len(comp) <= 1:
        return list(comp)
    
    # BFS from arbitrary node to find farthest
    start = comp[0]
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in adj[u]:
            if v in comp_set and v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    
    # BFS from farthest
    dist2 = {farthest: 0}
    parent2 = {farthest: None}
    q = deque([farthest])
    farthest2 = farthest
    while q:
        u = q.popleft()
        if dist2[u] > dist2[farthest2]:
            farthest2 = u
        for v in adj[u]:
            if v in comp_set and v not in dist2:
                dist2[v] = dist2[u] + 1
                parent2[v] = u
                q.append(v)
    
    # Reconstruct path
    path = []
    cur = farthest2
    while cur is not None:
        path.append(cur)
        cur = parent2[cur]
    path.reverse()
    return path

def _greedy_warnsdorff(adj, start, comp_set, n, deadline):
    """Warnsdorff-like greedy: prefer neighbors with fewest unvisited neighbors."""
    path = [start]
    visited = {start}
    while time.time() < deadline:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb in comp_set and nb not in visited]
        if not cands:
            break
        # Warnsdorff: prefer nodes with fewer unvisited neighbors
        cands.sort(key=lambda x: (len(adj[x] & comp_set - visited), x))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
    return path

def _greedy_randomized(adj, start, comp_set, n, deadline, rng):
    """Randomized greedy: pick from top-k candidates randomly."""
    path = [start]
    visited = {start}
    while time.time() < deadline:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb in comp_set and nb not in visited]
        if not cands:
            break
        # Score candidates
        scored = [(len(adj[x] & comp_set - visited), x) for x in cands]
        scored.sort()
        # Pick from top 3 with some randomness
        k = min(3, len(scored))
        idx = rng.randint(0, k - 1)
        nxt = scored[idx][1]
        path.append(nxt)
        visited.add(nxt)
    return path

def _extend_path(adj, path, comp_set, n, deadline):
    """Try to extend path from both ends greedily."""
    improved = True
    path_set = set(path)
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try extending the end
        end = path[-1]
        cands = [nb for nb in adj[end] if nb in comp_set and nb not in path_set]
        if cands:
            cands.sort(key=lambda x: (len(adj[x] & comp_set - path_set), x))
            path.append(cands[0])
            path_set.add(cands[0])
            improved = True
            continue
        # Try extending the beginning
        start_v = path[0]
        cands = [nb for nb in adj[start_v] if nb in comp_set and nb not in path_set]
        if cands:
            cands.sort(key=lambda x: (len(adj[x] & comp_set - path_set), x))
            path.insert(0, cands[0])
            path_set.add(cands[0])
            improved = True
            continue
    return path

def _insert_vertex(adj, path, vertex, comp_set):
    """Try to insert vertex into the path. Returns new path or None."""
    path_set = set(path)
    if vertex in path_set:
        return None
    # Check if vertex can be inserted at the start
    if vertex in adj[path[0]]:
        return [vertex] + path
    # Check if vertex can be inserted at the end
    if vertex in adj[path[-1]]:
        return path + [vertex]
    # Check internal insertion
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    return None

def _try_join_paths(adj, path1, path2, comp_set):
    """Try to join two paths. Returns combined path or None."""
    # Try: end of path1 connects to start of path2
    if path1[-1] in adj[path2[0]]:
        return path1 + path2
    # Try: end of path1 connects to end of path2 (reverse path2)
    if path1[-1] in adj[path2[-1]]:
        return path1 + path2[::-1]
    # Try: start of path1 connects to end of path2
    if path1[0] in adj[path2[-1]]:
        return path1[::-1] + path2
    # Try: start of path1 connects to start of path2
    if path1[0] in adj[path2[0]]:
        return path1[::-1] + path2[::-1]
    return None

def _dfs_exact(adj, comp, comp_set, n, deadline):
    """
    Exact DFS for finding longest path in small components.
    Uses Warnsdorff heuristic for ordering.
    """
    if len(comp) <= 1:
        return list(comp)
    
    best_path = []
    best_len = 0
    
    # Sort vertices by degree for better pruning
    comp_list = sorted(comp, key=lambda x: len(adj[x] & comp_set))
    
    def dfs(node, path, visited):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        path.append(node)
        visited.add(node)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # If we've found a Hamiltonian path, stop
        if len(path) == len(comp):
            path.pop()
            visited.remove(node)
            return
        
        # Get unvisited neighbors
        cands = [nb for nb in adj[node] if nb in comp_set and nb not in visited]
        
        if not cands:
            path.pop()
            visited.remove(node)
            return
        
        # Sort by Warnsdorff heuristic
        cands.sort(key=lambda x: len(adj[x] & comp_set - visited))
        
        for nb in cands:
            if time.time() > deadline:
                break
            dfs(nb, path, visited)
        
        path.pop()
        visited.remove(node)
    
    # Try from each vertex
    for start in comp_list:
        if time.time() > deadline:
            break
        dfs(start, [], set())
    
    return best_path

def _iterative_improvement(adj, path, comp_set, n, deadline, rng):
    """
    Iterative improvement: try insertions, 2-opt reversals, and re-extensions.
    """
    current = path[:]
    current_set = set(current)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to insert any unvisited vertex
        for v in sorted(comp_set - current_set, key=lambda x: len(adj[x] & comp_set - current_set)):
            new_path = _insert_vertex(adj, current, v, comp_set)
            if new_path is not None:
                current = new_path
                current_set = set(current)
                improved = True
                break
        
        if improved:
            current = _extend_path(adj, current, comp_set, n, deadline)
            current_set = set(current)
            continue
        
        # Phase 2: Try 2-opt style reversals to create new endpoints
        found = False
        for i in range(1, len(current) - 1):
            if time.time() > deadline:
                break
            
            # Reverse segment [0..i]
            # New start: current[i], new end: current[0]
            # Check if this creates valid path
            # Original: ... current[i-1] - current[i] - current[i+1] ...
            # After reverse [0..i]: current[i] - current[i-1] - ... - current[0] - current[1] - ...
            # Need: current[0] connected to current[1] (already true since it was a path)
            # Need: current[i] connected to current[i-1] (already true)
            # The reversal is always valid for a path!
            
            new_path = current[:i+1][::-1] + current[i+1:]
            new_set = set(new_path)
            
            # Try extending the new path
            extended = _extend_path(adj, new_path, comp_set, n, deadline)
            
            if len(extended) > len(current):
                current = extended
                current_set = set(current)
                found = True
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 3: Try reversing a middle segment
        found = False
        for i in range(1, len(current) - 2):
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                
                # Reverse segment [i..j]
                # Need: current[i-1] connected to current[j] AND current[i] connected to current[j+1] (if exists)
                if current[i-1] in adj[current[j]]:
                    if j + 1 < len(current) and current[i] in adj[current[j+1]]:
                        new_path = current[:i] + current[i:j+1][::-1] + current[j+1:]
                        new_set = set(new_path)
                        extended = _extend_path(adj, new_path, comp_set, n, deadline)
                        if len(extended) > len(current):
                            current = extended
                            current_set = set(current)
                            found = True
                            improved = True
                            break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 4: Try randomized restarts from different starts
        if time.time() > deadline - 5:
            break
        
        # Try a few random starts
        for _ in range(5):
            if time.time() > deadline:
                break
            start = rng.choice(list(comp_set))
            new_path = _greedy_warnsdorff(adj, start, comp_set, n, deadline)
            new_path = _extend_path(adj, new_path, comp_set, n, deadline)
            if len(new_path) > len(current):
                current = new_path
                current_set = set(current)
                improved = True
                break
        
        if not improved:
            # Try one more round of insertions with different ordering
            for v in sorted(comp_set - current_set, key=lambda x: len(adj[x] & current_set)):
                new_path = _insert_vertex(adj, current, v, comp_set)
                if new_path is not None:
                    current = new_path
                    current_set = set(current)
                    improved = True
                    break
            
            if not improved:
                break
    
    return current

def construct():
    n = P['n']
    adj = _build_adj()
    
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    import random
    rng = random.Random(42)
    
    # Find components
    components = _find_components(adj, n)
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in components:
        if len(comp) <= len(best_path):
            continue
        
        comp_set = set(comp)
        edge_count = sum(len(adj[u]) for u in comp) // 2
        
        if edge_count == len(comp) - 1:
            # Tree: use BFS diameter
            path = _bfs_longest_path_tree(adj, comp, comp_set)
            if len(path) > len(best_path):
                best_path = path
            continue
        
        # Cyclic component
        if time.time() > deadline:
            break
        
        if len(comp) <= 20:
            # Small enough for exact DFS
            bb_path = _dfs_exact(adj, comp, comp_set, n, deadline)
            if len(bb_path) > len(best_path):
                best_path = bb_path
        else:
            # Larger component: multi-start greedy + iterative improvement
            best_comp_path = []
            
            # Multi-start greedy
            for start in comp[:min(20, len(comp))]:
                if time.time() > deadline:
                    break
                path = _greedy_warnsdorff(adj, start, comp_set, n, deadline)
                path = _extend_path(adj, path, comp_set, n, deadline)
                if len(path) > len(best_comp_path):
                    best_comp_path = path
            
            # Also try randomized
            for _ in range(10):
                if time.time() > deadline:
                    break
                start = rng.choice(comp)
                path = _greedy_randomized(adj, start, comp_set, n, deadline, rng)
                path = _extend_path(adj, path, comp_set, n, deadline)
                if len(path) > len(best_comp_path):
                    best_comp_path = path
            
            # Iterative improvement
            best_comp_path = _iterative_improvement(adj, best_comp_path, comp_set, n, deadline, rng)
            
            if len(best_comp_path) > len(best_path):
                best_path = best_comp_path
    
    # Final extension
    if best_path:
        full_comp_set = set()
        for comp in components:
            if any(v in set(best_path) for v in comp):
                full_comp_set = set(comp)
                break
        if full_comp_set:
            best_path = _extend_path(adj, best_path, full_comp_set, n, deadline)
    
    # If no path found, return a single vertex
    if len(best_path) == 0:
        best_path = [0]
    
    # Verify validity
    path_set = set(best_path)
    assert len(path_set) == len(best_path), "Path has repeated vertices"
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in adj[best_path[i]], f"Invalid edge at position {i}"
    
    return best_path
# EVOLVE-BLOCK-END
