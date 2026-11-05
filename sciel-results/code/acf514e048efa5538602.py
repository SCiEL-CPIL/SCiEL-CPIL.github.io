"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _verify_girth(n, edges, k, g, max_time=10.0):
    """Verify girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    start_time = time.time()
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do non-backtracking BFS to find shortest cycle through it
    # A non-backtracking walk of length L from v back to v with no immediate backtracking
    # gives a cycle of length L (if L >= 3 and the walk is simple in the non-backtracking sense)
    # Actually, we need: no cycle of length < g.
    # Non-backtracking BFS: state is (vertex, prev_vertex). Start from each neighbor of v.
    
    for v in range(n):
        if time.time() - start_time > max_time:
            return False  # Timeout, assume invalid to be safe
        # Non-backtracking BFS from v
        # visited[prev][curr] = distance (number of edges in the non-backtracking walk)
        # We want to find the shortest non-backtracking closed walk from v
        # which corresponds to a cycle.
        
        # State: (current, previous) where previous is the vertex we came from
        # Start: for each neighbor u of v, state (u, v) at distance 1
        # If we reach state (v, w) where w != v, that's a closed non-backtracking walk of length dist
        # But we need to ensure it's a simple cycle. For girth checking, the shortest
        # non-backtracking closed walk from v has length equal to the girth if girth is odd,
        # or girth/2 * 2... Actually, let's use the standard method:
        # 
        # The girth is the length of the shortest cycle. In a bipartite graph, all cycles
        # are even. So we need no cycle of length < g (g is even for bipartite).
        # 
        # Non-backtracking BFS: from v, expand to (u, v) for u in N(v), dist=1.
        # From (u, v), expand to (w, u) for w in N(u) \ {v}, dist=2.
        # If we reach (v, w) for some w, the walk length is dist, and this forms a cycle
        # of length dist (since the graph is simple and non-backtracking, this is a valid cycle).
        # Wait, the cycle length is dist because we started at v, went to u (1 edge), 
        # ... and returned to v (1 edge). The total edges in the closed walk is dist.
        
        from collections import deque
        visited = {}  # (curr, prev) -> dist
        queue = deque()
        for u in adj[v]:
            visited[(u, v)] = 1
            queue.append((u, v))
        
        while queue:
            curr, prev = queue.popleft()
            dist = visited[(curr, prev)]
            if dist >= g:
                break  # We've already reached distance g, no shorter cycle
            for w in adj[curr]:
                if w == prev:
                    continue  # Non-backtracking
                if w == v:
                    # Found a cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                    # else continue, but we can break since we found girth >= g from this v
                    # Actually, we should check if dist+1 < g
                    if dist + 1 < g:
                        return False
                    break
                state = (w, curr)
                if state not in visited:
                    visited[state] = dist + 1
                    queue.append(state)
        # If we didn't find a cycle shorter than g from v, continue to next v
        # But we need to check all v's. However, if the graph is connected and we've
        # checked from one v and found no short cycle, it might still have one elsewhere.
        # Actually, the shortest cycle in the graph will be found from any vertex on it.
        # So we need to check all vertices.
    
    return True

def _luw_construct(k, d, q=None):
    """Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k.
    Returns (n, edges) or None if invalid."""
    if q is None:
        q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    elif q == 2:
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: x & y
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Rule table: rule[c] = (a, b) for c = 2..d
    # Standard LUW rule
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
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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
    
    n = len(order)
    return n, edges

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    Returns (n, edges)."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_n = float('inf')
    best_edges = None
    
    # The LUW graph CD(d, q) has girth at least 2d+2 for appropriate rules.
    # We need girth >= g, so we need 2d+2 >= g, i.e., d >= (g-2)/2.
    # For the standard LUW construction, the girth is exactly 2d+2 when d is odd,
    # and at least 2d+2 when d is even (depending on the field and rules).
    
    # Try different values of d
    min_d = max(3, (g - 2) // 2)
    if (g - 2) % 2 != 0:
        min_d += 1  # Ensure 2*min_d + 2 >= g
    
    # Try d = min_d, min_d+1, min_d+2, ... up to a reasonable limit
    # Each increment in d increases the number of vertices roughly by a factor of q
    # but also increases the girth. We want the smallest n.
    
    q = k
    # For q=5, d=3 gives girth ~8, d=4 gives girth ~10, etc.
    # We need girth >= g, so start with the smallest d that can achieve g.
    
    candidates = []
    for d in range(min_d, min_d + 5):
        if time.time() > deadline - 10.0:
            break
        try:
            n, edges = _luw_construct(k, d, q)
            if n < best_n:
                # Verify girth
                if _verify_girth(n, edges, k, g, max_time=5.0):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
    
    # If no valid graph found, try the default d from the original code
    if best_edges is None:
        d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            n, edges = _luw_construct(k, d_orig, q)
            best_n = n
            best_edges = edges
        except Exception:
            # Fallback: return a trivial valid graph if possible
            # For k=5, g=9, we need a 5-regular graph with girth >= 9
            # The LUW construction should work
            pass
    
    if best_edges is None:
        # Last resort: return an empty graph (invalid, but better than crashing)
        return 0, []
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
