"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import collections

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_all(n, adj):
    """BFS from every node. Returns list of distance lists, or None if disconnected."""
    all_dists = []
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
            return None
        all_dists.append(dist)
    return all_dists

def _diameter_from_dists(all_dists):
    return max(max(d) for d in all_dists)

def _is_triangle_free(n, edge_set):
    """Check triangle-free using adjacency sets."""
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
    """Full validation."""
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
    adj = [set() for _ in range(n)]
    for (u, v) in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    # Triangle-free
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    # Connected and diameter
    all_dists = _bfs_all(n, adj)
    if all_dists is None:
        return False
    return _diameter_from_dists(all_dists) <= 4

def _construct():
    """
    Find minimum added edges to make path diameter <= 4 while triangle-free.
    
    Strategy: The path P(24) has diameter 23. We need to add shortcuts.
    Key insight: A triangle-free graph with 24 vertices and diameter <= 4.
    
    Approach: Use a structured construction based on a "wheel-like" or "grid" pattern
    that naturally avoids triangles, then verify and optimize.
    
    Let's think about what structure works:
    - If we partition vertices into groups and connect groups in a cycle-like fashion,
      we can avoid triangles.
    - A bipartite graph is always triangle-free.
    
    For a path 0-1-2-...-23, the even vertices {0,2,...,22} and odd vertices {1,3,...,23}
    form a natural bipartition. The path itself is bipartite.
    
    If we add edges only within the same parity group, we break bipartiteness but might
    create triangles. If we add edges only between even and odd (cross edges), we stay
    bipartite and thus triangle-free.
    
    So: add cross-parity edges (even-to-odd) to reduce diameter.
    
    The complete bipartite graph K_{12,12} has diameter 2. But we don't need all edges.
    We need enough cross edges so that the diameter is <= 4.
    
    Let's try a structured approach:
    - Keep the path edges.
    - Add a set of cross-parity edges that create short paths between all pairs.
    
    One effective structure: connect each vertex to vertices at distance 4 and 5 (or similar)
    in the path, ensuring they're of opposite parity.
    
    Actually, let me think more carefully. The path distance between i and j is |i-j|.
    We need to reduce this to <= 4 for all pairs.
    
    A good approach: add edges (i, i+k) for various k, where i and i+k have opposite parity.
    Since i and i+k have opposite parity iff k is odd, we add edges with odd offsets.
    
    Let's try adding edges with offset 5 (odd, so cross-parity):
    (0,5), (1,6), (2,7), ..., (18,23)
    That's 19 edges. With these, can we get diameter <= 4?
    
    Let's think: from 0, we can go 0->5->10->15->20->23 (using offset 5 edges and path).
    Actually 0->5 is direct, 5->10 is direct, etc. So 0 to 20 is 4 steps via 5,10,15,20.
    0 to 23: 0->5->10->15->20->23, that's 5 steps. Not enough.
    
    Let's also add offset 3 edges: (0,3), (1,4), ..., (20,23)
    That's 21 more edges. Total 40.
    
    With offsets 3 and 5:
    From 0: 0->3->6->9->12->15->18->21->23 (using 3s and then 2)
    That's still long.
    
    Better: use a mix. Let me try a different approach.
    
    Actually, the minimum number of edges to add to a path of n vertices to get diameter d
    is a known problem. For n=24, d=4, triangle-free constraint.
    
    Let me try a concrete construction:
    Add edges to create a "ladder" or "grid" pattern.
    
    Alternative: Use the fact that we can add any cross-parity edges.
    Let's add edges that create a 2D grid-like structure.
    
    Map vertex i to (i//6, i%6) or similar, and add edges within rows and columns.
    
    Actually, let me just do a smart greedy with the constraint that we only add
    cross-parity edges (to guarantee triangle-free).
    """
    n = P['n']
    base = [(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']]
    base_set = set(base)
    
    adj = [set() for _ in range(n)]
    for (u, v) in base:
        adj[u].add(v)
        adj[v].add(u)
    
    added = []
    edge_set = set(base)
    
    # Only consider cross-parity edges (guarantees triangle-free since base is bipartite)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set and u % 2 != v % 2:
                candidates.append((u, v))
    
    deadline = time.time() + 50
    
    # Greedy: repeatedly add the edge that most reduces the diameter
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        all_dists = _bfs_all(n, adj)
        if all_dists is None:
            break
        diam = _diameter_from_dists(all_dists)
        if diam <= 4:
            break
        
        # Find best edge
        best_edge = None
        best_diam = diam + 1
        
        for (u, v) in candidates:
            if (u, v) in edge_set:
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
            new_dists = _bfs_all(n, adj)
            if new_dists is not None:
                new_diam = _diameter_from_dists(new_dists)
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
            
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.discard((u, v))
        
        if best_edge is None or best_diam >= diam:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        added.append([u, v])
        candidates.remove((u, v))
    
    return added

def construct():
    result = _construct()
    if _validate(P['n'], P['edges'], result):
        return [[int(u), int(v)] for u, v in result]
    
    # Fallback: add all cross-parity non-base edges
    base = set(tuple(e) for e in P['edges'])
    result = []
    for u in range(P['n']):
        for v in range(u + 1, P['n']):
            if u % 2 != v % 2 and (u, v) not in base:
                result.append([u, v])
    return [[int(u), int(v)] for u, v in result]
# EVOLVE-BLOCK-END
