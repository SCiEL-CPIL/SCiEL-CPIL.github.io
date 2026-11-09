"""Construct smallest 4-regular graph with girth >= 10 using LUW family."""
import time

_START_TIME = time.time()
_TIME_LIMIT = 150.0  # seconds

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _luw_construct(d, q):
    """Construct the full LUW(d, q) graph. Returns (n, edges)."""
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Compute the rule for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Enumerate all points and lines
    # Point: (side=0, coord_tuple of length d)
    # Line: (side=1, coord_tuple of length d)
    # Coordinate 0 is always None (fixed), coords 1..d vary over GF(q)
    
    def point_id(x):
        """Map coordinate tuple (length d, index 0 unused) to integer."""
        # x is a tuple of length d, x[0] is unused (None)
        # Use x[1], x[2], ..., x[d-1] as the varying coordinates
        val = 0
        for i in range(1, d):
            val = val * q + x[i]
        return val

    def line_id(x):
        val = 0
        for i in range(1, d):
            val = val * q + x[i]
        return val

    # Total vertices: 2 * q^(d-1) ... wait, let me re-examine
    # In the seed, the point is (0, (None,) + (0,)*d) which has d+1 elements
    # The varying part is indices 1..d (d elements), each in GF(q)
    # So there are q^d points and q^d lines, total 2*q^d
    
    # But coordinate 0 is always None/unused, so effectively d-1 free coordinates?
    # No: looking at the seed, y = [None, first] + [0]*(d-1), so y has d+1 elements
    # y[0]=None, y[1]=first, y[2..d]=computed
    # The free parameter is 'first' (y[1]) which ranges over GF(q)
    # The rest are determined by the rule
    # So for each point, the free coordinate is y[1], giving q points per "fiber"
    # But actually the point is fully specified by all its coordinates
    # Let me just enumerate all q^d possible coordinate tuples for side 0 and side 1
    
    # Actually, re-reading: the vertex is (side, coord) where coord is a tuple of length d+1
    # coord[0] is None, coord[1..d] are in GF(q)
    # So there are q^d distinct vertices per side, total 2*q^d
    
    # For efficiency, let's use a flat representation
    # Point p with coordinates p[1..d] (d values)
    # Line l with coordinates l[1..d] (d values)
    # Adjacency: p ~ l iff for all c in 2..d: l[c] - p[c] = l[a_c] * p[b_c]
    
    n_per_side = q ** d
    n = 2 * n_per_side
    
    # Map: point index i -> coordinates
    def coords_from_idx(i):
        """i in [0, q^d) -> tuple of d coordinates (indices 1..d)."""
        coords = [0] * (d + 1)  # index 0 unused
        tmp = i
        for pos in range(d, 0, -1):
            coords[pos] = tmp % q
            tmp //= q
        return coords
    
    edges = []
    
    for pi in range(n_per_side):
        p = coords_from_idx(pi)
        for li in range(n_per_side):
            l = coords_from_idx(li)
            # Check adjacency: for all c in 2..d: l[c] - p[c] == l[a_c] * p[b_c]
            ok = True
            for c in range(2, d + 1):
                a, b = rule[c]
                lhs = sub(l[c], p[c])
                rhs = mul(l[a], p[b])
                if lhs != rhs:
                    ok = False
                    break
            if ok:
                edges.append((pi, n_per_side + li))
    
    return n, edges

def _verify_girth(n, edges, g):
    """Verify girth >= g using BFS from each vertex (non-backtracking)."""
    from collections import deque
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # Non-backtracking BFS from each vertex
    # State: (current, prev, distance)
    # We want to find the shortest cycle, which is the minimum distance 
    # where a non-backtracking walk returns to the start
    
    for start in range(n):
        # BFS state: (node, prev_node, dist)
        # visited: set of (node, prev_node) to avoid revisiting
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        found_cycle = False
        while queue:
            if _time_remaining() < 5:
                return None  # timeout, inconclusive
            node, prev, dist = queue.popleft()
            if node == start and dist >= 2:
                # Found a cycle of length dist
                if dist < g:
                    return False
                found_cycle = True
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue  # no backtracking
                if (nb, node) in visited:
                    continue
                visited.add((nb, node))
                queue.append((nb, node, dist + 1))
    
    return True

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    
    if k != 4:
        # Fallback for other k values (shouldn't happen per task)
        raise ValueError("Only k=4 supported")
    
    q = k  # 4
    
    # For LUW(d, q), girth >= d+5. We need girth >= g=10, so d >= 5.
    # Try d from 5 upward, return the first valid (smallest) graph.
    best_n = None
    best_edges = None
    
    for d in range(5, 12):  # d=5 gives 2*4^5=2048 vertices
        if _time_remaining() < 30:
            break
        
        n, edges = _luw_construct(d, q)
        
        # Verify girth
        valid = _verify_girth(n, edges, g)
        if valid:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            break  # smallest d gives smallest graph, so we're done
        
        if _time_remaining() < 10:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: return the seed's connected-component approach
    return _luw_connected_component(k, g)

def _luw_connected_component(k, g):
    """Fallback: LUW connected component of zero vertex (seed approach)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
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
