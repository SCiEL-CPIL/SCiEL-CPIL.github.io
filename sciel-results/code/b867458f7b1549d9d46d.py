"""
Longest path in a sparse G(100, 1.8/100) graph via structural decomposition:
- Trees: exact diameter via two BFS.
- Unicyclic: try removing each cycle edge, take max tree diameter.
- Multi-cyclic: DP over subsets for small components, heuristic DFS for larger.
"""
import random
from collections import defaultdict, deque

def _build_graph():
    n = 100
    rng = random.Random(748)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 1.8 / 100:
                edges.add((i, j))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

N, ADJ = _build_graph()

def _connected_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

COMPS = _connected_components()

def _bfs_longest_path(adj, start, blocked_edge=None):
    """BFS from start, returning (length, path) of longest path found.
    If blocked_edge is (u,v), that edge is not traversed."""
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    dist[start] = 0
    q = deque([start])
    best_node = start
    best_dist = 0
    while q:
        u = q.popleft()
        if dist[u] > best_dist:
            best_dist = dist[u]
            best_node = u
        for v in adj[u]:
            if blocked_edge and ((u == blocked_edge[0] and v == blocked_edge[1]) or
                                  (u == blocked_edge[1] and v == blocked_edge[0])):
                continue
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    # Reconstruct path
    path = [best_node]
    cur = best_node
    while parent[cur] != -1:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return best_dist + 1, path

def _tree_diameter(adj, nodes, idx_map):
    """Find diameter of a tree component. adj is local adjacency (list of sets of local indices)."""
    n = len(adj)
    if n == 0:
        return 1, [0]
    if n == 1:
        return 1, [0]
    # Pick arbitrary start
    start = 0
    # BFS to find farthest
    _, far1_path = _bfs_longest_path(adj, start)
    far1 = far1_path[-1]
    # BFS from far1
    length, path = _bfs_longest_path(adj, far1)
    # Map back to global
    global_path = [nodes[i] for i in path]
    return length, global_path

def _find_cycle_in_unicyclic(adj, n):
    """Find the cycle in a unicyclic graph. Returns list of local indices in cycle order."""
    # Use DFS to find back edge
    visited = [False] * n
    parent = [-1] * n
    cycle = []
    
    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                if dfs(v):
                    return True
            elif v != parent[u]:
                # Found back edge u-v, reconstruct cycle
                cycle.append(v)
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.reverse()
                return True
        return False
    
    for i in range(n):
        if not visited[i]:
            if dfs(i):
                break
    
    if cycle:
        # Remove duplicates if any
        seen = set()
        unique_cycle = []
        for v in cycle:
            if v not in seen:
                seen.add(v)
                unique_cycle.append(v)
        return unique_cycle
    return []

def _longest_path_unicyclic(adj, nodes, idx_map):
    """For unicyclic component, try removing each cycle edge, compute tree diameter."""
    n = len(adj)
    if n <= 2:
        if n == 1:
            return 1, [nodes[0]]
        if n == 2 and 1 in adj[0]:
            return 2, [nodes[0], nodes[1]]
        return 1, [nodes[0]]
    
    cycle = _find_cycle_in_unicyclic(adj, n)
    if not cycle:
        # Fallback to tree diameter
        return _tree_diameter(adj, nodes, idx_map)
    
    best_len = 0
    best_path = [nodes[0]]
    
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        # Remove edge (u, v)
        adj[u].discard(v)
        adj[v].discard(u)
        
        length, path = _tree_diameter(adj, nodes, idx_map)
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        if length > best_len:
            best_len = length
            best_path = path
    
    return best_len, best_path

def _dp_longest_path(adj, n):
    """DP over subsets for small components. Returns (length, path)."""
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [0]
    
    size = 1 << n
    dp = [bytearray(n) for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = 1
    
    best_len = 1
    best_mask = 1
    best_last = 0
    
    for mask in range(size):
        for last in range(n):
            if not dp[mask][last]:
                continue
            for nxt in adj[last]:
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                dp[new_mask][nxt] = 1
                if new_mask.bit_count() > best_len:
                    best_len = new_mask.bit_count()
                    best_mask = new_mask
                    best_last = nxt
    
    # Reconstruct
    path = [best_last]
    mask = best_mask
    last = best_last
    while mask.bit_count() > 1:
        prev_mask = mask ^ (1 << last)
        found = False
        for prev in range(n):
            if prev_mask & (1 << prev) and dp[prev_mask][prev] and prev in adj[last]:
                path.append(prev)
                mask = prev_mask
                last = prev
                found = True
                break
        if not found:
            break
    path.reverse()
    return best_len, path

def _heuristic_longest_path(adj, nodes, idx_map):
    """Heuristic DFS for larger multi-cyclic components."""
    n = len(adj)
    best_path = [0]
    
    for start in range(n):
        path = [start]
        used = {start}
        
        def dfs(current):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
            remaining = n - len(used)
            if len(path) + remaining <= len(best_path):
                return
            for nxt in adj[current]:
                if nxt not in used:
                    used.add(nxt)
                    path.append(nxt)
                    dfs(nxt)
                    path.pop()
                    used.discard(nxt)
        
        dfs(start)
    
    return len(best_path), [nodes[i] for i in best_path]

def _longest_path_in_component(comp):
    n = len(comp)
    if n == 0:
        return []
    if n == 1:
        return [comp[0]]
    
    idx_map = {v: i for i, v in enumerate(comp)}
    adj = [set() for _ in range(n)]
    edge_count = 0
    for u in comp:
        for v in ADJ[u]:
            if v in idx_map:
                adj[idx_map[u]].add(idx_map[v])
                edge_count += 1
    edge_count //= 2
    
    if edge_count == n - 1:
        # Tree
        _, path = _tree_diameter(adj, comp, idx_map)
        return path
    elif edge_count == n:
        # Unicyclic
        _, path = _longest_path_unicyclic(adj, comp, idx_map)
        return path
    else:
        # Multi-cyclic
        if n <= 18:
            _, local_path = _dp_longest_path(adj, n)
            return [comp[i] for i in local_path]
        else:
            _, path = _heuristic_longest_path(adj, comp, idx_map)
            return path

component_paths = []
for comp in COMPS:
    lp = _longest_path_in_component(comp)
    if lp:
        component_paths.append(lp)

if component_paths:
    BEST_PATH = max(component_paths, key=len)
else:
    BEST_PATH = [0]

def construct():
    return list(BEST_PATH)
