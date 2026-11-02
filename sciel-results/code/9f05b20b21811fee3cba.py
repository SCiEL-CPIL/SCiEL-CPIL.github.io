"""Improved program: Lazebnik-Ustimenko-Woldar graph with optimized construction."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    # For the specific task: k=5, g=9
    # Use Lazebnik-Ustimenko-Woldar construction
    # q = k = 5 (prime)
    # d is chosen such that girth >= d + 5 (for odd d)
    # We need girth 9, so d + 5 >= 9, d >= 4
    # For odd d: d = 5 gives girth >= 10 (better)
    # For even d: d = 4 gives girth >= 8 (not enough)
    # Actually, LUW: girth is at least d+5 if d is odd, d+4 if d is even
    # We need girth >= 9, so:
    #   odd d: d+5 >= 9 => d >= 4, smallest odd is d=5
    #   even d: d+4 >= 9 => d >= 5, smallest even is d=6
    # d=5 gives q^(d+1) = 5^6 = 15625 vertices (too many)
    # Wait, the LUW graph has q^(d+1) vertices? No.
    # Actually LUW graph CD(d,q) has 2*q^(d+1) vertices? Let me reconsider.
    
    # The current seed program returns the connected component of zero.
    # Let me just use the same approach but verify correctness.
    
    q = k
    # Choose d to minimize vertices while ensuring girth >= g
    # LUW: girth >= d + 5 for odd d, girth >= d + 4 for even d
    # We need girth >= g
    best_d = None
    best_vertices = float('inf')
    
    for d_candidate in range(3, 12):
        if d_candidate % 2 == 1:
            girth_min = d_candidate + 5
        else:
            girth_min = d_candidate + 4
        if girth_min >= g:
            n_vertices = q ** (d_candidate + 1)
            if n_vertices < best_vertices:
                best_vertices = n_vertices
                best_d = d_candidate
    
    if best_d is None:
        best_d = 5  # fallback
    
    d = best_d
    n_expected = q ** (d + 1)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        mod_q = q
        def add(x, y): return (x + y) % mod_q
        def sub(x, y): return (x - y) % mod_q
        def mul(x, y): return x * y % mod_q
    
    # Build the rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Vertices are (side, y) where side in {0,1} and y is tuple of length d+1
    # y[0] is ignored (None), y[1]..y[d] are field elements
    # Actually from the seed: vertex = (side, x) where x = (None, first, 0, ..., 0)
    # Let me trace through the seed code more carefully.
    
    # In the seed:
    # y = [None, first] + [0] * (d - 1)  -> length d+1, indices 0..d
    # y[0] = None, y[1] = first, y[2..d] = 0
    # Then for c in 2..d: y[c] = ...
    # So the tuple has d+1 elements, index 0 is None, indices 1..d are values
    
    # Neighbors: for each 'first' in 0..q-1, compute the neighbor
    def get_neighbors(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    # BFS from start to find connected component
    start_vertex = (0, (None,) + (0,) * d)
    
    # Encode vertices for efficiency
    def encode(v):
        side, x = v
        # x[0] is None, x[1..d] are values
        return (side, x[1])
    
    index = {}
    order = []
    visited = set()
    
    start_enc = encode(start_vertex)
    index[start_enc] = 0
    order.append(start_vertex)
    visited.add(start_enc)
    
    queue = deque([0])
    
    while queue:
        vi = queue.popleft()
        if time.time() > deadline:
            break
        vertex = order[vi]
        for nb in get_neighbors(vertex):
            nb_enc = encode(nb)
            if nb_enc not in visited:
                visited.add(nb_enc)
                index[nb_enc] = len(order)
                order.append(nb)
                queue.append(len(order) - 1)
    
    n = len(order)
    
    # Build edge list (only from side 0 vertices to avoid duplicates)
    edges = []
    for vi in range(n):
        if time.time() > deadline:
            break
        vertex = order[vi]
        if vertex[0] != 0:
            continue
        for nb in get_neighbors(vertex):
            nb_enc = encode(nb)
            if nb_enc in index:
                edges.append((vi, index[nb_enc]))
    
    return n, edges
# EVOLVE-BLOCK-END
