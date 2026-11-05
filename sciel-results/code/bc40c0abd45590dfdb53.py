"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    construction family, trying multiple parameterizations to minimize n.
    
    The LUW construction CD(d, q) gives a q-regular bipartite graph with girth >= d+5 (odd d)
    or d+4 (even d). We try various d values that yield girth >= g, and pick the one with
    fewest vertices in the connected component of the zero point.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    q = k  # field size (must be prime or 4)
    
    # For LUW: girth is at least d+5 for odd d, d+4 for even d
    # We need girth >= g
    # So for odd d: d >= g - 5, for even d: d >= g - 4
    # d must be at least 3
    
    # Try all valid d values and pick the one giving fewest vertices
    best_n = None
    best_edges = None
    
    # Determine range of d to try
    # For odd d: d >= g-5, and d must be odd
    # For even d: d >= g-4, and d must be even
    # We'll try d from the minimum up to some reasonable upper bound
    # Larger d means more vertices, so we prefer smaller d, but need to verify girth
    
    # Minimum d for each parity
    min_d_odd = g - 5
    if min_d_odd < 3:
        min_d_odd = 3
    if min_d_odd % 2 == 0:
        min_d_odd += 1
    
    min_d_even = g - 4
    if min_d_even < 3:
        min_d_even = 3
    if min_d_even % 2 == 1:
        min_d_even += 1
    
    # Try d values; typically the smallest valid d gives the fewest vertices
    # But we should verify girth exactly
    candidates = []
    if min_d_odd >= 3:
        candidates.append(min_d_odd)
        candidates.append(min_d_odd + 2)
    if min_d_even >= 3:
        candidates.append(min_d_even)
        candidates.append(min_d_even + 2)
    
    # Also try d = g-5 if odd, g-4 if even (the tightest)
    candidates = sorted(set(c for c in candidates if c >= 3))
    
    for d in candidates:
        if time.time() > deadline - 5.0:
            break
        try:
            n, edges = _luw_construct(q, d)
            if n is not None and n > 0:
                # Verify girth >= g using non-backtracking walks
                if _verify_girth(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
        except Exception:
            continue
    
    if best_n is None:
        # Fallback: use the original approach with a safe d
        d = min_d_odd if min_d_odd >= 3 else 3
        if d % 2 == 0:
            d += 1
        n, edges = _luw_construct(q, d)
        return n, edges
    
    return best_n, best_edges


def _luw_construct(q, d):
    """Build the LUW graph CD(d, q) and return the connected component of the zero point."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # q must be prime for the standard construction
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Compute the rule for each column c = 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _verify_girth(n, edges, g):
    """Verify that the graph has no cycle shorter than g using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS and check that the shortest cycle through it is >= g
    # A cycle of length L through vertex v means there's a path of length L-1 from v back to v
    # More precisely: for each vertex v, the girth is the minimum over all pairs (u,v) of 
    # the shortest path length + 1 (if u and v share a common neighbor)
    
    # Efficient approach: for each vertex, BFS up to depth floor((g-1)/2) and check
    # if any two neighbors at the same level are the same (which would indicate a short cycle)
    
    half = (g - 1) // 2
    
    for start in range(n):
        # BFS from start, tracking parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
# EVOLVE-BLOCK-END
