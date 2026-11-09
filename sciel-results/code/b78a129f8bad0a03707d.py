import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency structure once
_ADJ = [set() for _ in range(P['n'])]
_EDGE_SET = set()
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)
    _EDGE_SET.add((min(u, v), max(u, v)))

def _is_edge(u, v):
    return (min(u, v), max(u, v)) in _EDGE_SET

def _max_matching_hopcroft_karp():
    """
    Find maximum matching in the original graph using Hopcroft-Karp.
    Returns the matching as a list of (u, v) pairs.
    """
    n = P['n']
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in _ADJ[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found
    
    def dfs(u):
        for v in _ADJ[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    
    # Extract matching edges
    match_edges = []
    for u in range(n):
        if pair_u[u] != -1:
            match_edges.append((u, pair_u[u]))
    return match_edges

def _matching_to_path(match_edges):
    """
    Convert a matching to a path by connecting matched edges.
    A matching gives us disjoint edges; we try to chain them into a path.
    """
    if not match_edges:
        return []
    
    # Build the graph from matching edges only
    match_adj = {v: [] for v in range(P['n'])}
    for u, v in match_edges:
        match_adj[u].append(v)
        match_adj[v].append(u)
    
    # Find all components that are paths (each vertex has degree 0 or 1 in matching)
    visited = set()
    paths = []
    
    for start in range(P['n']):
        if start in visited or not match_adj[start]:
            continue
        # Traverse the path
        path = [start]
        visited.add(start)
        current = start
        while match_adj[current]:
            nxt = match_adj[current][0]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            current = nxt
    
        if len(path) > 1:
            paths.append(path)
    
    # Sort paths by length descending
    paths.sort(key=lambda p: -len(p))
    
    # Try to connect paths using original graph edges
    # Start with the longest path
    if not paths:
        return []
    
    best_path = list(paths[0])
    remaining = paths[1:]
    
    # Try to extend best_path from both ends using other path segments
    changed = True
    while changed and remaining:
        changed = False
        for i, seg in enumerate(remaining):
            # Try to attach seg to front or back of best_path
            # Check if seg[0] or seg[-1] connects to best_path[0] or best_path[-1]
            connected = False
            
            # Try front: seg[-1] -> best_path[0]
            if _is_edge(seg[-1], best_path[0]):
                best_path = seg + best_path
                connected = True
            
            # Try back: best_path[-1] -> seg[0]
            if not connected and _is_edge(best_path[-1], seg[0]):
                best_path = best_path + seg
                connected = True
            
            # Try reversed: seg[0] -> best_path[0]
            if not connected and _is_edge(seg[0], best_path[0]):
                best_path = list(reversed(seg)) + best_path
                connected = True
            
            # Try reversed back: best_path[-1] -> seg[-1]
            if not connected and _is_edge(best_path[-1], seg[-1]):
                best_path = best_path + list(reversed(seg))
                connected = True
            
            if connected:
                remaining.pop(i)
                changed = True
                break
    
    return best_path

def _greedy_path(start, time_limit):
    """Greedy path construction using Warnsdorff-like heuristic."""
    path = [start]
    seen = {start}
    while True:
        candidates = _ADJ[path[-1]] - seen
        if not candidates:
            break
        # Choose candidate with fewest onward options (Warnsdorff)
        v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def _extend_both_ends(path, time_limit):
    """Try to extend a path from both ends greedily."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending front
        if path:
            front = path[0]
            candidates = _ADJ[front] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        # Try extending back
        if path:
            back = path[-1]
            candidates = _ADJ[back] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def _local_search(path, deadline, max_iters=50):
    """
    Local search to improve a path by trying 2-opt and 3-opt style moves.
    """
    path = list(path)
    n = len(path)
    if n < 3:
        return path
    
    best = path[:]
    best_len = n
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random 2-opt: remove two consecutive edges and reconnect
        if n >= 4:
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Current: ... path[i-1] - path[i] - ... - path[j-1] - path[j] - path[j+1] ...
            # Try: ... path[i-1] - path[j] ... path[i] - path[j+1] ...
            # But we need to check if edges exist
            
            a, b = path[i-1], path[i]
            c, d = path[j-1], path[j]
            e = path[j+1] if j + 1 < n else None
            
            # Check if we can flip the segment
            # New connections: a-c and b-e (or a-d and b-... depending on orientation)
            
            # Try reversal of segment [i..j]
            new_path = path[:i] + list(reversed(path[i:j+1])) + path[j+1:]
            
            # Validate all edges
            valid = True
            for k in range(len(new_path) - 1):
                if not _is_edge(new_path[k], new_path[k+1]):
                    valid = False
                    break
            
            if valid and len(set(new_path)) == len(new_path):
                if len(new_path) > best_len:
                    best = new_path
                    best_len = len(new_path)
                    path = new_path
                    n = len(path)
        
        # Try inserting an unvisited vertex into the path
        visited = set(path)
        unvisited = [v for v in range(P['n']) if v not in visited]
        if unvisited:
            # Try to insert a random unvisited vertex
            v = rng.choice(unvisited)
            for pos in range(len(path) + 1):
                # Check if we can insert v at position pos
                if pos > 0 and not _is_edge(path[pos-1], v):
                    continue
                if pos < len(path) and not _is_edge(v, path[pos]):
                    continue
                
                new_path = path[:pos] + [v] + path[pos:]
                if len(set(new_path)) == len(new_path):
                    if len(new_path) > best_len:
                        best = new_path
                        best_len = len(new_path)
                        path = new_path
                        break
    
    return best

def _backtrack_path(start, max_depth, time_limit, deadline):
    """
    Depth-limited backtracking search for a long simple path starting at 'start'.
    Uses Warnsdorff ordering to prune efficiently.
    Returns the longest path found within max_depth and time budget.
    """
    best_path = [start]
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = list(path)
        if depth >= max_depth:
            return
        
        current = path[-1]
        candidates = _ADJ[current] - seen
        if not candidates:
            return
        
        # Order by Warnsdorff: fewest onward options first
        candidates = sorted(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
        
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    best_path = []
    
    # Phase 0: Get initial lower bound from maximum matching
    match_edges = _max_matching_hopcroft_karp()
    match_path = _matching_to_path(match_edges)
    if len(match_path) > len(best_path):
        best_path = match_path
    
    # Phase 1: Greedy from every start vertex
    for start in range(P['n']):
        if time.time() > deadline:
            break
        path = _greedy_path(start, deadline)
        path = _extend_both_ends(path, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Phase 2: Local search on best path
    if len(best_path) >= 3:
        best_path = _local_search(best_path, deadline, max_iters=100)
    
    # Phase 3: Backtracking search from promising starts
    # Use the best greedy path's endpoints and high-degree vertices as seeds
    seed_vertices = set()
    if best_path:
        seed_vertices.add(best_path[0])
        seed_vertices.add(best_path[-1])
    
    # Add high-degree vertices
    for v in range(P['n']):
        if len(_ADJ[v]) >= 4:
            seed_vertices.add(v)
    
    # Sort seeds by degree descending for better initial exploration
    seed_vertices = sorted(seed_vertices, key=lambda v: -len(_ADJ[v]))
    
    for start in seed_vertices:
        if time.time() > deadline:
            break
        # Try increasing depth limits
        for max_depth in [15, 25, 40, 60]:
            if time.time() > deadline:
                break
            path = _backtrack_path(start, max_depth, deadline, deadline)
            if len(path) > len(best_path):
                best_path = path
                # Local search on improved path
                if len(best_path) >= 3:
                    best_path = _local_search(best_path, deadline, max_iters=50)
                # Early exit if we found a Hamiltonian path
                if len(best_path) == P['n']:
                    return best_path
    
    # Phase 4: Randomized restarts with greedy + backtracking hybrid
    rng = random.Random(748)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        start = rng.randint(0, P['n'] - 1)
        path = _greedy_path(start, deadline)
        path = _extend_both_ends(path, deadline)
        
        # If greedy path is short, try backtracking from its endpoint
        if len(path) < len(best_path) + 5:
            for endpoint in [path[0], path[-1]]:
                if time.time() > deadline:
                    break
                bp = _backtrack_path(endpoint, 50, deadline, deadline)
                if len(bp) > len(best_path):
                    best_path = bp
                    if len(best_path) >= 3:
                        best_path = _local_search(best_path, deadline, max_iters=30)
        
        if len(path) > len(best_path):
            best_path = path
            if len(best_path) >= 3:
                best_path = _local_search(best_path, deadline, max_iters=30)
    
    # Final: try to extend best_path from both ends one more time
    if best_path:
        best_path = _extend_both_ends(best_path, deadline)
        if len(best_path) >= 3:
            best_path = _local_search(best_path, deadline, max_iters=100)
    
    return best_path
# EVOLVE-BLOCK-END
