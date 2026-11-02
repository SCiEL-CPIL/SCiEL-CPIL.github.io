<<<<<<< SEARCH
def _verify_bipartite_luw(n, edges, k, g, deadline):
    """Faster verification for bipartite LUW graphs.
    LUW graphs are bipartite with parts of size n/2.
    For a bipartite graph, all cycles are even.
    For even girth g, we need no cycle of length < g.
    We can do BFS from one side only and check distances to the other side.
    """
    if time.time() > deadline:
        return False

    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)

    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # LUW graphs are bipartite: side 0 vertices are 0..n/2-1, side 1 are n/2..n-1
    # For even girth g, we need no cycle shorter than g.
    # In a bipartite graph, a cycle of length 2r means there are two distinct
    # non-backtracking walks of length r from some vertex to another.
    # Equivalently: BFS from each vertex on side 0, and check that no vertex
    # on side 1 is reached at distance < g/2 via two different paths.
    
    half_n = n // 2
    # For even girth g, max cycle length to check is g-1 (which is odd, so no cycle)
    # Actually for bipartite, cycles are even. So we check for cycles of length 4, 6, ..., g-2
    # A cycle of length 2r exists iff there's a vertex v on side 0 and a vertex w on side 1
    # such that there are >= 2 non-backtracking walks of length r from v to w.
    # Simpler: BFS from each side-0 vertex, track for each side-1 vertex how many
    # distinct predecessors at the final level.
    
    # Actually the simplest correct check: non-backtracking BFS from every vertex,
    # but we can halve the work by noting the bipartite structure.
    # Let's just do the standard check but with a set for visited (faster than dict).
    
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= g:
                continue
            if node == start and dist >= 2:
                if dist < g:
                    return False
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    if dist + 1 < g:
                        queue.append((nb, node, dist + 1))

    return True
=======
def _verify_bipartite_luw(n, edges, k, g, deadline):
    """Faster verification for bipartite LUW graphs."""
    return _verify_graph_fast(n, edges, k, g, deadline)
>>>>>>> REPLACE
