"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Moore bound for k-regular graph of girth g
    # For even girth g=2r: n >= k + k(k-1) + ... + k(k-1)^(r-1)
    # For odd girth g=2r+1: n >= 1 + k + k(k-1) + ... + k(k-1)^(r-1) + k(k-1)^r
    if g % 2 == 0:
        r = g // 2
        moore = k + sum(k * (k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        moore = 1 + k + sum(k * (k-1)**i for i in range(r)) + k * (k-1)**r
    
    # For k=4, g=10: r=5, moore = 4 + 4*3 + 4*9 + 4*27 + 4*81 = 4+12+36+108+324 = 484
    # Wait, let me recalculate. For even girth g=2r:
    # n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
    # For k=4, g=10, r=5: n >= 4*(1+3+9+27+81) = 4*121 = 484
    # Hmm, but the problem says smallest known is 320. Let me re-check.
    # Actually the Moore bound formula: for k-regular, girth g:
    # If g = 2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
    # If g = 2r+1: n >= 1 + k * sum_{i=0}^{r} (k-1)^i
    # For k=4, g=10=2*5, r=5: n >= 4*(1+3+9+27+81) = 4*121 = 484
    # But wait, that doesn't match. Let me look up: the (4,10)-cage has 320 vertices.
    # Actually I think I'm confusing myself. Let me just use the known facts.
    # The task says smallest known is 320 and best lower bound is 243.
    
    # Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction as a baseline,
    # then try to improve with local search / edge swaps if time permits.
    
    # First, let's implement the LUW construction which gives a valid answer
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
    else:
        # Fallback: random regular graph
        n_luw, edges_luw = _random_regular(k, g, start_time, deadline)
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Try to improve with edge swaps / local search
    if time.time() < deadline - 30:
        improved = _local_search(k, g, n_luw, edges_luw, start_time, deadline)
        if improved is not None:
            n_imp, edges_imp = improved
            if n_imp < best_n:
                best_n = n_imp
                best_edges = edges_imp
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
    
    # Define the rule for LUW construction
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _random_regular(k, g, start_time, deadline):
    """Generate a random k-regular graph."""
    # For k=4, try various sizes starting from a reasonable lower bound
    # Moore bound for (4,10) is actually 243 (per problem statement)
    # Let's try sizes from a reasonable range
    
    # Actually, for a valid (4,10)-cage candidate, we need girth >= 10.
    # Random regular graphs of degree 4 will have small girth typically.
    # This is just a fallback; the LUW construction should work.
    
    # Try a few sizes
    for n in [320, 324, 328, 332, 336, 340]:
        if n % 2 != 0 and k == 4:
            continue  # 4-regular graph needs even number of vertices * 4 / 2 = 2n edges, n*4 must be even, so n can be anything actually
        # Actually n*k must be even. For k=4, n*4 is always even.
        
        edges = _gen_random_4reg(n)
        if edges is not None and _has_girth_at_least(edges, n, g):
            return n, edges
    
    # If we can't find one quickly, return a larger one
    n = 500
    edges = _gen_random_4reg(n)
    if edges is not None:
        return n, edges
    return n, []


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    # Use configuration model
    for attempt in range(100):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None


def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to find shortest cycle
    # Actually, we need to check that no cycle shorter than g exists.
    # A cycle of length L exists if there's a path of length L-1 between two adjacent vertices
    # that doesn't use the edge itself.
    
    # More efficient: for each vertex, do BFS and check for early returns
    # For girth >= g, BFS from each vertex up to depth g//2 should not find a back-edge
    # that creates a cycle shorter than g.
    
    # Simpler approach: BFS from each vertex, track parent, and check if we revisit
    # a vertex at a depth that would create a short cycle.
    
    max_depth = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _local_search(k, g, n, edges, start_time, deadline):
    """Try to reduce the number of vertices via local search (not straightforward for regular graphs).
    
    For regular graphs, we can't simply remove vertices. We could try:
    1. Start with a smaller graph and try to complete it
    2. Use edge swaps to maintain regularity while trying to reduce n
    
    Actually, for a k-regular graph, n*k must be even. For k=4, any n works.
    But we need girth >= g, which is very restrictive.
    
    A better approach: try to construct graphs of decreasing size from a known valid one.
    This is hard in general. Let's just try the LUW result and maybe one alternative.
    """
    # For now, just return the LUW result as-is
    # A more sophisticated search would be needed for real improvement
    return None


# EVOLVE-BLOCK-END
