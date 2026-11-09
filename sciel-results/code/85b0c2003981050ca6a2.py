import time
import sys
from collections import deque

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320.
    # We search for a circulant graph C(n, {s1, s2})
    
    # Pre-compute LUW result as a fallback in case search takes too long or fails
    # But LUW is slow to compute if we do it inside the loop. 
    # We will just run the search. If it fails, we call _luw.
    
    # To ensure we don't timeout, we limit the search range.
    # n from 246 to 600.
    # Steps s1, s2.
    
    for n in range(246, 600):
        if time.time() - t_start > _DEADLINE - 5:
            break
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 5:
                break
            for s2 in range(s1 + 1, max_step + 1):
                # Quick 4-cycle check
                if (s1 + s2) % n == 0:
                    continue
                if (s2 - s1) % n == 0: # s1 == s2 mod n, handled by s2 > s1
                    continue
                # Check girth
                if _check_girth_circulant(n, s1, s2, 10):
                    edges = []
                    for u in range(n):
                        # Add edges u -> u+s1 and u -> u+s2
                        # To avoid duplicates, only add if u < v
                        v1 = (u + s1) % n
                        v2 = (u + s2) % n
                        if u < v1:
                            edges.append((u, v1))
                        if u < v2:
                            edges.append((u, v2))
                    return n, edges
                    
    # If no circulant found, fallback to LUW
    return _luw(k, g)

def _check_girth_circulant(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    Girth < g iff there exists a non-backtracking walk of length < g that returns to start.
    Actually, girth is the length of the shortest cycle.
    We can check by BFS from 0 up to depth g//2 - 1.
    If we encounter a vertex at depth d that was already visited at depth d' < d, 
    and the path doesn't backtrack, we have a cycle.
    More simply: 
    A cycle of length L exists if there is a sequence of steps $x_1, ..., x_L \in \{s1, -s1, s2, -s2\}$
    such that sum $x_i = 0$ and the partial sums are distinct (except start/end) and no backtracking.
    
    Efficient check:
    For each vertex v, count the number of vertices at distance d.
    In a graph of girth >= g, the number of vertices at distance d from v is exactly
    1 + k * sum_{i=0}^{d-1} (k-1)^i for d < g/2.
    For k=4, g=10:
    d=0: 1
    d=1: 4
    d=2: 4 * 3 = 12
    d=3: 12 * 3 = 36
    d=4: 36 * 3 = 108
    Total vertices within distance 4: 1 + 4 + 12 + 36 + 108 = 161.
    If n < 161, impossible. (246 > 161, so ok).
    
    We can just run BFS from 0. If the BFS tree has any "cross edge" or "back edge" 
    that creates a cycle shorter than g, it fails.
    Actually, if the number of nodes at distance d is less than expected, a cycle exists.
    """
    # Expected counts
    expected = [1, 4, 12, 36, 108]
    
    # BFS from 0
    visited = [-1] * n
    dist = [0] * n
    q = deque()
    q.append(0)
    visited[0] = 0
    dist[0] = 0
    
    counts = [0] * 5
    counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= 4:
            continue
        
        # Neighbors: u+s1, u-s1, u+s2, u-s2
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if visited[v] == -1:
                visited[v] = d + 1
                dist[v] = d + 1
                q.append(v)
                if d + 1 < 5:
                    counts[d + 1] += 1
            else:
                # If visited, check if it creates a short cycle
                # If v is at distance d, it's a back edge to parent (ok, tree edge)
                # If v is at distance d+1, it's a cross edge to a node at same level? 
                # If dist[v] == d+1, then u-v is an edge between nodes at same distance?
                # No, u is at d, v is at d+1. This is a tree edge.
                # If dist[v] <= d, then we have a cycle.
                # Cycle length = dist[u] + 1 + dist[v]?
                # If dist[v] == d, cycle length is 2*d + 1 (odd cycle).
                # If dist[v] < d, cycle length is dist[u] + 1 + dist[v].
                if dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    # We need to ensure this is a valid simple cycle.
                    # In BFS, an edge (u,v) where dist[v] <= dist[u] implies a cycle 
                    # of length dist[u] - dist[v] + 1? No.
                    # The cycle is path 0->...->v->...->u->v?
                    # Actually, if dist[v] == dist[u], cycle is even?
                    # Let's just check if the cycle length is < g.
                    if cycle_len < g:
                        return False
    
    # After BFS, check if counts match expected
    # If counts[d] < expected[d], it means some nodes were merged into shorter cycles
    for d in range(1, 5):
        if counts[d] < expected[d]:
            return False
            
    return True

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

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
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
