"""
Construct a 5-regular graph of girth 9 via the line graph of a 3-regular graph
with girth 14.

Key insight:
- The line graph L(G) of a 3-regular graph G is (2*3-2) = 4-regular, not 5.
- However, the DIRECTED line graph or a modified construction can work.
- Actually, for a 3-regular graph G, each vertex in L(G) corresponds to an edge
  in G. Two edges are adjacent in L(G) if they share a vertex. In a 3-regular
  graph, each edge is incident to 2 vertices, and at each vertex, 3 edges meet,
  so each edge is adjacent to (3-1) + (3-1) = 4 other edges. So L(G) is 4-regular.

This is a problem. Let me reconsider.

Alternative interpretation: Perhaps the intended construction uses a different
type of graph. Let me think about what gives a 5-regular graph with girth 9.

Actually, let me try a different approach: use a known construction or a
randomized search to find a 5-regular graph with girth at least 9.

Given the time constraint (120s), I'll implement a randomized greedy construction
with local search to maximize girth.

Strategy:
1. Start with a random 5-regular graph on n vertices (n chosen large enough).
2. Use local search (edge swaps) to eliminate short cycles.
3. Verify girth at the end.

For n, I'll start with a reasonable upper bound and try to minimize.
The Moore bound for k=5, g=9 (odd girth 2r+1 with r=4) is:
1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706
But the best known is 1116, so I'll try to find something in that range.

Actually, let me try a more structured approach using algebraic constructions
or known families of high-girth graphs.

Let me implement a randomized construction with careful girth checking.
"""

import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: use a randomized approach with local search.
    """
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # For k=5, g=9, try to construct a 5-regular graph with girth >= 9
    # using a randomized greedy approach with local search.
    
    start_time = time.time()
    time_limit = 110  # Leave some buffer for verification
    
    # Try different sizes, starting from a reasonable lower bound
    # The Moore bound is 1706, best known is 1116
    # Let's try to find something around 1200-1500 vertices
    
    best_n = None
    best_edges = None
    
    # Try a few different n values
    for n in [1200, 1300, 1400, 1500, 1600]:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        # Try to construct a 5-regular graph on n vertices with girth >= 9
        edges = _construct_high_girth(n, k, g, time_limit - (time.time() - start_time))
        
        if edges is not None:
            # Verify the graph is valid
            if _verify_graph(n, edges, k, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use LUW construction
    return _luw_construct(5, 9)


def _construct_high_girth(n, k, g, time_budget):
    """
    Try to construct a k-regular graph on n vertices with girth at least g
    using a randomized greedy approach with local search.
    """
    if n < 1:
        return None
    
    # First, generate a random k-regular graph
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Convert adjacency to edge list
    edges = _adj_to_edges(adj, n)
    
    # Local search: try to increase girth by swapping edges
    start_time = time.time()
    
    for iteration in range(100000):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Check current girth
        current_girth = _compute_girth(adj, n)
        if current_girth >= g:
            break
        
        # Try a random edge swap
        if _try_edge_swap(adj, n, k, g):
            # Update edges
            pass  # We'll recompute at the end
    
    # Final check
    current_girth = _compute_girth(adj, n)
    if current_girth < g:
        return None
    
    return _adj_to_edges(adj, n)


def _generate_random_regular(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    
    # Use configuration model
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    
    i = 0
    while i < len(half_edges):
        u = half_edges[i]
        j = i + 1
        while j < len(half_edges) and (half_edges[j] == u or adj[u][half_edges[j]]):
            j += 1
        if j >= len(half_edges):
            return None  # Failed to create a simple graph
        v = half_edges[j]
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
        # Remove the used half-edges
        half_edges[i] = half_edges[-1]
        half_edges.pop()
        half_edges[j] = half_edges[-1]
        half_edges.pop()
        # Adjust indices
        if j > i:
            j -= 1
        if i >= len(half_edges):
            break
        i += 1
    
    # Verify all degrees are k
    for v in range(n):
        actual_deg = sum(adj[v])
        if actual_deg != k:
            return None
    
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append((u, v))
    return edges


def _compute_girth(adj, n):
    """Compute the girth of a graph using BFS from each vertex."""
    min_girth = float('inf')
    
    for start in range(n):
        # BFS to find shortest cycle through start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u][v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth <= 2:
                            return 2
    
    return min_girth if min_girth != float('inf') else n + 1


def _try_edge_swap(adj, n, k, target_girth):
    """
    Try to perform an edge swap that increases girth.
    Returns True if a swap was made.
    """
    # Find a random edge to remove
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append((u, v))
    
    if not edges:
        return False
    
    # Try a few random swaps
    for _ in range(100):
        u1, v1 = random.choice(edges)
        
        # Find another edge to swap with
        u2, v2 = random.choice(edges)
        if u1 == u2 and v1 == v2:
            continue
        if u1 == v2 or v1 == u2 or u1 == v1 or u2 == v2:
            continue
        
        # Check if the swap would create multi-edges or loops
        new_edges = [(u1, v2), (u2, v1)]
        if adj[u1][v2] or adj[u2][v1]:
            continue
        if u1 == v2 or u2 == v1:
            continue
        
        # Try the swap
        adj[u1][v1] = False
        adj[v1][u1] = False
        adj[u2][v2] = False
        adj[v2][u2] = False
        adj[u1][v2] = True
        adj[v2][u1] = True
        adj[u2][v1] = True
        adj[v1][u2] = True
        
        # Check if girth improved
        new_girth = _compute_girth(adj, n)
        if new_girth >= target_girth or new_girth > _compute_girth(adj, n):
            return True
        else:
            # Revert
            adj[u1][v2] = False
            adj[v2][u1] = False
            adj[u2][v1] = False
            adj[v1][u2] = False
            adj[u1][v1] = True
            adj[v1][u1] = True
            adj[u2][v2] = True
            adj[v2][u2] = True
    
    return False


def _verify_graph(n, edges, k, g):
    """Verify that the graph is k-regular and has girth at least g."""
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        if u == v or u >= n or v >= n:
            return False
        if adj[u][v]:
            return False  # Duplicate edge
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    
    # Check degrees
    for v in range(n):
        if deg[v] != k:
            return False
    
    # Check girth
    girth = _compute_girth(adj, n)
    return girth >= g


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
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
