"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter(n, adj):
    """Compute diameter via BFS from each node."""
    import collections
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = collections.deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
        max_d = max(max_d, max(dist))
    return max_d

def _is_triangle_free(n, edge_set):
    """Check if graph is triangle-free using adjacency sets."""
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True

def _validate(n, base_edges, added_edges):
    """Full validation: no duplicates, no existing edges, triangle-free, connected, diameter<=4."""
    all_edges = set()
    for e in base_edges:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    for e in added_edges:
        u, v = e
        if not (0 <= u < n and 0 <= v < n):
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in all_edges:
            return False
        all_edges.add(key)
    # Check triangle-free
    adj = [set() for _ in range(n)]
    for (u, v) in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    # Check connected and diameter
    import collections
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = collections.deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _construct_greedy():
    """Greedy construction: start with path, add edges to reduce diameter while keeping triangle-free."""
    n = P['n']
    base = [(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']]
    base_set = set(base)
    
    # Start with no added edges
    added = []
    edge_set = set(base)
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 50  # safety margin
    
    # Precompute candidate edges (not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    # Greedy: repeatedly find the edge that most reduces the diameter
    # while maintaining triangle-free property
    for iteration in range(200):
        if time.time() > deadline:
            break
        # Compute current diameter
        import collections
        diam = 0
        connected = True
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = collections.deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                connected = False
                break
            diam = max(diam, max(dist))
        
        if not connected or diam <= 4:
            break
        
        # Find best edge to add
        best_edge = None
        best_score = float('inf')
        for (u, v) in candidates:
            if (u, v) in edge_set:
                continue
            # Check triangle-free: no common neighbor
            common = adj[u] & adj[v]
            if common:
                continue
            # Add edge, compute new diameter
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
            new_diam = 0
            new_connected = True
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = collections.deque([src])
                while queue:
                    uu = queue.popleft()
                    for vv in adj[uu]:
                        if dist[vv] == -1:
                            dist[vv] = dist[uu] + 1
                            queue.append(vv)
                if -1 in dist:
                    new_connected = False
                    break
                new_diam = max(new_diam, max(dist))
            
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.discard((u, v))
            
            if not new_connected:
                continue
            # Score: prefer edges that reduce diameter
            score = new_diam
            if score < best_score or (score == best_score and (best_edge is None or (u, v) < best_edge)):
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        added.append([u, v])
        candidates.remove((u, v))
    
    return added

def _construct_structured():
    """
    Construct a structured triangle-free graph with diameter <= 4.
    Use a bipartite-like structure with strategic edges.
    """
    n = P['n']
    base = [(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']]
    base_set = set(base)
    
    # Strategy: Create a graph where vertices are connected through a few "hub" structures
    # that maintain triangle-free property.
    # 
    # Approach: Use a 4-partite structure or a modified ladder.
    # Actually, let's try: connect vertices i to i+4 and i to i+5 (if triangle-free).
    
    # Let's try a specific construction:
    # Add edges to create a "4-step" connectivity pattern.
    # For a path, adding edges (i, i+k) for appropriate k can reduce diameter.
    
    # Triangle-free constraint: if we add (u,v), we can't have any w adjacent to both.
    
    # Let's try: add all edges (i, i+4) where i+4 < n, and (i, i+5) where i+5 < n,
    # but check triangle-freeness.
    
    added = []
    edge_set = set(base)
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    def try_add(u, v):
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        if u == v:
            return False
        # Check triangle-free
        if adj[u] & adj[v]:
            return False
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        return True
    
    # Try adding edges in a structured way
    # First, add long-range edges that don't create triangles
    for offset in [4, 5, 6, 3, 7]:
        for i in range(n - offset):
            if time.time() > deadline_global:
                break
            try_add(i, i + offset)
        if time.time() > deadline_global:
            break
    
    # Check if diameter is <= 4
    import collections
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = collections.deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist or max(dist) > 4:
            # Need more edges, fall back to greedy
            return None
    
    return added

deadline_global = time.time() + 50

def construct():
    global deadline_global
    deadline_global = time.time() + 55
    
    # Try structured construction first
    result = _construct_structured()
    if result is not None:
        # Validate
        if _validate(P['n'], P['edges'], result):
            return [[int(u), int(v)] for u, v in result]
    
    # Fall back to greedy
    result = _construct_greedy()
    if _validate(P['n'], P['edges'], result):
        return [[int(u), int(v)] for u, v in result]
    
    # Last resort: use the original approach (all odd-distance edges)
    base = set(tuple(e) for e in P['edges'])
    result = [[u, v] for u in range(P['n']) for v in range(u + 1, P['n']) if (u - v) % 2 and (u, v) not in base]
    return [[int(u), int(v)] for u, v in result]
# EVOLVE-BLOCK-END
