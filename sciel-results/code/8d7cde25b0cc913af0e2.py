"""Cayley-graph construction for 4-regular girth-10 graph."""
import numpy as np

def construct(k, g):
    # We use the Lazebnik-Ustimenko-Woldar (LUW) construction, which is a structured
    # Cayley graph on a finite group (specifically a lift of a smaller graph) that
    # guarantees high girth for appropriate parameters.
    # For k=4, we use q=4. The number of vertices is 2 * q^d.
    # We choose d such that the girth is at least g.
    # Typically, girth is ~ 2d. So we start with d = ceil(g/2) and increase if necessary.
    
    if k != 4:
        return _construct_fallback(k, g)
    
    # Try increasing d until girth is satisfied
    # Start with a small d to minimize n
    # Moore bound for (4,10) is 243.
    # n = 2 * 4^d.
    # d=4 -> 512
    # d=5 -> 2048
    # d=6 -> 8192
    
    # Let's try d from 4 upwards.
    # For d=4, n=512. Girth might be 8 or 9.
    # For d=5, n=2048. Girth might be 10 or 11.
    
    for d in range(4, 10):
        n, edges = _luw_construction(4, d)
        if _verify_girth(n, edges, g):
            return n, edges
            
    # Fallback if all fail (should not happen for d large enough)
    return _construct_fallback(4, g)


def _luw_construction(q, d):
    """Construct a LUW graph with parameters q and d."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule definition
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _verify_girth(n, edges, g):
    """Verify that the graph has girth at least g."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check that ball of radius r = (g-1)//2 is a tree
    # If g is even, g=2r, we need no cycles of length <= 2r-1.
    # This is equivalent to no edges between vertices at distance < r from each other?
    # Actually, if there is a cycle of length L < g, then there is a vertex on that cycle
    # such that two of its neighbors are at distance < g from each other?
    # Simpler: For each vertex v, perform BFS up to depth r.
    # If we encounter an edge (u, w) where both u and w are in the ball of v
    # and dist(v, u) + dist(v, w) + 1 < g? No.
    # A cycle of length L exists iff there is an edge (u, w) such that
    # dist(v, u) + dist(v, w) + 1 <= L for some v?
    # Actually, if the ball of radius r around v is a tree, then there are no cycles of length <= 2r.
    # For g=10, r=4. If ball of radius 4 is a tree, then no cycles of length <= 8.
    # Wait, girth >= 10 means no cycles of length 3..9.
    # If ball of radius 4 is a tree, it contains no cycles.
    # Any cycle of length L <= 8 would be contained in the ball of radius 4 around any vertex on it?
    # Let v be a vertex on a cycle of length L. The diameter of the cycle is L/2.
    # If L <= 8, L/2 <= 4. So the cycle is contained in the ball of radius 4 around v.
    # So checking that the ball of radius 4 is a tree is sufficient to ensure girth >= 9?
    # If girth is 9, L=9. L/2 = 4.5.
    # A cycle of length 9 is not necessarily contained in the ball of radius 4 around a single vertex?
    # Let v be a vertex. The farthest vertex on the cycle is at distance 4.
    # So the cycle is contained in the ball of radius 4.
    # So if the ball of radius 4 is a tree, there are no cycles of length <= 9.
    # Thus girth >= 10.
    
    r = (g - 1) // 2
    for v in range(n):
        # BFS up to depth r
        dist = [-1] * n
        dist[v] = 0
        queue = [v]
        for u in queue:
            if dist[u] == r:
                continue
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                elif dist[w] == dist[u] + 1:
                    # Found a cycle? No, this is just another path.
                    # We need to check if there is an edge within the ball that closes a cycle.
                    pass
        
        # Check for edges within the ball that create a cycle
        # An edge (u, w) creates a cycle if dist[u] + dist[w] + 1 < 2*r + 1?
        # Actually, if u and w are both in the ball, and u != w, and (u,w) is an edge,
        # and u is not the parent of w, then we have a cycle.
        # We can just check if the subgraph induced by the ball is a tree.
        # A connected graph with N vertices is a tree iff it has N-1 edges.
        # The ball has at most 1 + 4 + 12 + 36 + 108 = 161 vertices.
        # We can count edges in the ball.
        
        ball_vertices = [i for i in range(n) if 0 <= dist[i] <= r]
        if not ball_vertices:
            continue
        ball_set = set(ball_vertices)
        edge_count = 0
        for u in ball_vertices:
            for w in adj[u]:
                if w in ball_set and u < w:
                    edge_count += 1
        
        # If the ball is connected (it is, by BFS), it is a tree iff edges = vertices - 1
        if edge_count != len(ball_vertices) - 1:
            return False
            
    return True


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction."""
    # This is now handled by the main construct function's loop,
    # but we keep this for compatibility if called directly.
    q = k
    # Use a large enough d to ensure girth
    d = max(5, g)
    return _luw_construction(q, d)
