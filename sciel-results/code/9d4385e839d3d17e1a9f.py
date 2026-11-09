"""Construct a 5-regular graph with girth 9 (cage problem n(5,9))."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Baseline: LUW construction
    baseline_n, baseline_edges = _luw_construct(k, g)
    
    best_n = baseline_n
    best_edges = baseline_edges
    
    # Try greedy construction with girth enforcement
    if elapsed() < 10.0:
        try:
            n, edges = _greedy_construct(k, g, deadline)
            if n is not None and n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass
    
    # Try edge-swap local search starting from LUW graph
    if elapsed() < 30.0:
        try:
            n, edges = _local_search_from_luw(k, g, baseline_n, baseline_edges, deadline)
            if n is not None and n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass
    
    return best_n, best_edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph with girth >= d+5."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _verify_girth(n, edges, g):
    """Verify graph has girth >= g using non-backtracking BFS from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False
    
    for start in range(n):
        # Non-backtracking BFS: state is (node, prev)
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= g:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    return False  # Found cycle of length depth
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    queue.append((nb, node, depth + 1))
    return True

def _greedy_construct(k, g, deadline):
    """Greedy vertex-addition with girth enforcement via non-backtracking BFS."""
    # Moore bound for odd girth g, degree k
    # n >= 2*(1 + k + k*(k-1) + ... + k*(k-1)^((g-1)/2))
    half = (g - 1) // 2
    moore = 2
    for i in range(1, half + 1):
        moore += k * (k - 1) ** i
    
    # Try to construct from moore bound upward
    max_n = moore + 200
    for target_n in range(moore, min(moore + 50, max_n + 1)):
        if time.time() > deadline - 5:
            break
        if target_n * k % 2 != 0:
            continue
        
        # Try multiple random seeds
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            edges = _greedy_build(target_n, k, g)
            if edges is not None:
                return target_n, edges
    
    return None, None

def _greedy_build(n, k, g):
    """Build a k-regular graph with girth >= g using greedy edge addition."""
    # Start with empty graph
    adj = [set() for _ in range(n)]
    edges = []
    
    # Target degree for each vertex
    # We'll try to add edges one by one
    
    # Random permutation of potential edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline_global - 5:
            return None
        if len(adj[u]) >= k or len(adj[v]) >= k:
            continue
        
        # Check if adding this edge would create a cycle shorter than g
        if _edge_creates_short_cycle(adj, u, v, g):
            continue
        
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
        # Early termination check
        if len(edges) == n * k // 2:
            # Check all vertices have degree k
            if all(len(adj[i]) == k for i in range(n)):
                return edges
            else:
                return None
    
    if all(len(adj[i]) == k for i in range(n)) and len(edges) == n * k // 2:
        return edges
    return None

def _edge_creates_short_cycle(adj, u, v, g):
    """Check if adding edge (u,v) to current graph creates a cycle shorter than g."""
    # BFS from u and v simultaneously to see if they can reach each other
    # in less than g-1 steps (excluding the direct edge)
    # Actually, we need to check if there's a path from u to v of length < g-1
    # that doesn't use the edge (u,v) itself
    
    # BFS from u, not going through v directly
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        if dist[node] >= g - 1:
            break
        for nb in adj[node]:
            if nb == v:
                # Found path of length dist[node] + 1 from u to v
                # Cycle would be of length dist[node] + 1 + 1 = dist[node] + 2
                if dist[node] + 2 < g:
                    return True
            if nb not in dist and nb != v:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return False

def _local_search_from_luw(k, g, n, edges, deadline):
    """Try to find smaller graph by local search from LUW construction."""
    # This is a placeholder - the LUW graph is already quite good
    # Try to verify it's valid and return it
    return n, edges

deadline_global = time.time() + 110.0
