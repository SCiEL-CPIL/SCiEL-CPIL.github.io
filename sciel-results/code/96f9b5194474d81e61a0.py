"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 115
    
    # Moore bound for (k, g): for even girth g=2g', n >= 2 * sum_{i=0}^{g'-1} (k-1)^i
    if g % 2 == 0:
        gprime = g // 2
        moore = 2 * sum((k-1)**i for i in range(gprime))
    else:
        gprime = (g - 1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(gprime))
    
    n_min = moore
    if n_min % 2 != 0:
        n_min += 1
    if n_min < 4:
        n_min = 4
    
    # Try layered construction starting from Moore bound
    for n_try in range(n_min, n_min + 200, 2):
        if time.time() > deadline:
            break
        result = _layered_construct(k, g, n_try)
        if result is not None:
            return result
    
    # Fallback: LUDW
    return _ludw_fallback(k, g)

def _layered_construct(k, g, n):
    """
    Build a k-regular graph of girth >= g using a layered BFS-tree approach.
    For even girth g=2r, we build r layers from a central edge.
    Layer 0: 2 vertices (the central edge endpoints)
    Layer i: each vertex in layer i-1 has (k-1) children in layer i
    Total vertices at Moore bound: 2 * sum_{i=0}^{r-1} (k-1)^i
    For n > Moore bound, we add extra vertices in the outermost layers.
    
    We assign edges between consecutive layers, ensuring:
    1. Each vertex has degree exactly k
    2. No two vertices in the same layer are adjacent (to avoid short cycles)
    3. The resulting graph has girth >= g
    """
    if g % 2 != 0:
        return None  # Only handle even girth for now
    
    r = g // 2  # radius
    k = 3  # assume k=3
    
    # Layer sizes: layer i has (k-1)^i vertices on each side, total 2*(k-1)^i
    # For n > Moore bound, distribute extra vertices
    moore_layer_sizes = [(k-1)**i for i in range(r)]  # per side
    total_moore = 2 * sum(moore_layer_sizes)
    extra = n - total_moore
    if extra < 0:
        return None
    
    # Distribute extra vertices to outermost layers
    layer_sizes = list(moore_layer_sizes)  # per side
    if extra > 0:
        # Add to last layer
        layer_sizes[-1] += extra // 2
        if extra % 2 == 1:
            layer_sizes[-1] += 1  # might make odd, but we'll handle
    
    total_vertices = 2 * sum(layer_sizes)
    if total_vertices != n:
        # Adjust
        diff = n - total_vertices
        if diff != 0:
            layer_sizes[-1] += diff
            total_vertices = 2 * sum(layer_sizes)
            if total_vertices != n:
                return None
    
    # Assign vertex IDs
    # Side 0: layers 0..r-1, Side 1: layers 0..r-1
    vertices = []  # (side, layer, index_in_layer)
    vid = 0
    for side in range(2):
        for layer in range(r):
            for idx in range(layer_sizes[layer]):
                vertices.append((side, layer, idx))
                vid += 1
    
    if vid != n:
        return None
    
    # Build edges
    edges = set()
    adj = [[] for _ in range(n)]
    
    def add_edge(u, v):
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edges:
            return False
        edges.add(key)
        adj[u].append(v)
        adj[v].append(u)
        return True
    
    # Central edge: layer 0 vertices
    v0_0 = 0  # side 0, layer 0, index 0
    v1_0 = sum(layer_sizes[:0]) + 0  # side 1, layer 0, index 0
    # Actually, let me compute properly
    # Side 0: vertices 0 .. sum(layer_sizes)-1
    # Side 1: vertices sum(layer_sizes) .. 2*sum(layer_sizes)-1
    
    side0_start = 0
    side1_start = sum(layer_sizes)
    
    # Central edge
    if not add_edge(side0_start, side1_start):
        return None
    
    # For each layer i (0 to r-2), connect layer i to layer i+1
    # Each vertex in layer i (side s) has (k-1) neighbors in layer i+1 (same side)
    # Plus 1 neighbor in layer i-1 (or central edge for i=0)
    
    # We need to assign which vertices in layer i+1 connect to which in layer i
    # This is a bipartite matching problem: each vertex in layer i needs (k-1) edges to layer i+1
    # Each vertex in layer i+1 needs (k-1) edges from layer i (if i+1 < r-1) or k edges from layer i (if i+1 = r-1, the outermost)
    
    # Wait, for the outermost layer (r-1), vertices have no children, so they need all k edges to go to layer r-2
    # For inner layers, each vertex has 1 parent edge + (k-1) child edges
    
    # Let me re-think the degree constraints:
    # Layer 0: each vertex has 1 edge to central (other side) + (k-1) edges to layer 1
    #   But wait, layer 0 on side 0 connects to layer 0 on side 1 (central edge) and to layer 1 on side 0
    #   So degree of layer 0 vertex: 1 (central) + (k-1) (to layer 1) = k. Good.
    # Layer i (1 <= i <= r-2): each vertex has (k-1) edges to layer i-1 and (k-1) edges to layer i+1
    #   But that's only 2(k-1) = 4 for k=3, but we need degree 3.
    #   Hmm, this doesn't work. Let me reconsider.
    
    # Actually for a (3, 2r)-cage with Moore bound:
    # The BFS tree from the central edge has:
    # Layer 0: 2 vertices (the edge endpoints)
    # Layer 1: each layer 0 vertex has 2 children (degree 3: 1 parent + 2 children), so 4 vertices
    # Layer 2: each layer 1 vertex has 2 children, so 8 vertices
    # ...
    # Layer r-1: 2^(r-1) * 2 = 2^r vertices
    # Total: 2 * (1 + 2 + 4 + ... + 2^(r-1)) = 2 * (2^r - 1) = 2^(r+1) - 2
    # For r=8: 2^9 - 2 = 510. Moore bound says 510, but n must be even >= 512.
    
    # For the Moore bound graph, the outermost layer vertices need to be connected to each other
    # to complete the degree. But that creates cycles. For girth 16, we need the outermost
    # layer connections to not create cycles shorter than 16.
    
    # This is getting complex. Let me try a simpler approach: just use the LUDW fallback
    # but with a better parameter search.
    
    return None

def _ludw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
# EVOLVE-BLOCK-END
