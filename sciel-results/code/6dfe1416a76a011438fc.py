<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite
    graph whose girth is at least d + 5 for odd d. Returns the connected component of the
    zero point. Optimized with integer-encoded vertices and precomputed rules."""
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    q = k
    # For girth g, we need d such that d+5 >= g, i.e., d >= g-5.
    # LW requires d odd for the girth guarantee d+5.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # ensure odd d for the standard LW girth bound

    # Precompute field operations for q=4 (GF(4) as F2^2) or prime q
    if q == 4:
        # GF(4): {0,1,2,3}, addition = XOR, multiplication via table
        MUL_TABLE = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def fadd(x, y): return x ^ y
        def fsub(x, y): return x ^ y  # in char 2, sub = add
        def fmul(x, y): return MUL_TABLE[x][y]
    else:
        def fadd(x, y): return (x + y) % q
        def fsub(x, y): return (x - y) % q
        def fmul(x, y): return (x * y) % q

    # Precompute the rule: for each c in 2..d, rule[c] = (a, b)
    # such that the incidence is l[c] - p[c] = l[a] * p[b]
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule_a[c], rule_b[c] = 1, c0 - 2
        elif r == 1:
            rule_a[c], rule_b[c] = c0 - 1, 1
        elif r == 2:
            rule_a[c], rule_b[c] = c0, 1
        else:
            rule_a[c], rule_b[c] = 1, c0 + 1

    # Encode vertices as integers: side * q^d + value
    # where value encodes (x1, x2, ..., xd) in base q
    # Side 0 = points, Side 1 = lines
    # For points: (None, x1, x2, ..., xd) -> we store x1..xd
    # For lines: (None, x1, x2, ..., xd) -> we store x1..xd

    # Precompute powers of q
    qpow = [1] * (d + 1)
    for i in range(1, d + 1):
        qpow[i] = qpow[i-1] * q

    def encode(side, coords):
        """coords is tuple of d integers (x1..xd)"""
        v = side * qpow[d]
        for i in range(d):
            v += coords[i] * qpow[i]
        return v

    def decode(val):
        side = val // qpow[d]
        val -= side * qpow[d]
        coords = [0] * d
        for i in range(d):
            coords[i] = val % q
            val //= q
        return side, coords

    # For efficiency, precompute the neighbor computation
    # A point p = (x1, ..., xd) is adjacent to line l = (y1, ..., yd) iff:
    #   y1 = x1 (first coordinate matches)
    #   For c = 2..d: y[c] - x[c] = y[rule_a[c]] * x[rule_b[c]]  (for point->line)
    #   Equivalently: x[c] = y[c] - y[rule_a[c]] * x[rule_b[c]]  ... wait let me re-derive

    # From the original code:
    # For point p (side=0), computing line l:
    #   y[1] = first (free parameter)
    #   y[c] = x[c] + y[a] * x[b]  for c=2..d  (since side==0: add(x[c], mul(y[a], x[b])))
    #
    # For line l (side=1), computing point p:
    #   y[1] = first (free parameter)
    #   y[c] = x[c] - x[a] * y[b]  for c=2..d  (since side==1: sub(x[c], mul(x[a], y[b])))

    # So for a point p with coords x (indices 0..d-1 representing x1..xd):
    #   For each first_val in 0..q-1:
    #     y[0] = first_val  (this is y1)
    #     For c_idx in 1..d-1 (representing c = c_idx+2):
    #       a = rule_a[c_idx+2] - 1  (0-indexed)
    #       b = rule_b[c_idx+2] - 1  (0-indexed)
    #       y[c_idx] = fadd(x[c_idx], fmul(y[a], x[b]))
    #   Line coords = y

    # For a line l with coords x (indices 0..d-1 representing x1..xd):
    #   For each first_val in 0..q-1:
    #     y[0] = first_val
    #     For c_idx in 1..d-1:
    #       a = rule_a[c_idx+2] - 1
    #       b = rule_b[c_idx+2] - 1
    #       y[c_idx] = fsub(x[c_idx], fmul(x[a], y[b]))
    #   Point coords = y

    # Start from the zero point: side=0, coords=(0,0,...,0)
    start_val = encode(0, tuple([0]*d))

    # BFS to find connected component
    from collections import deque
    index = {start_val: 0}
    order = [start_val]
    edges = []
    adj = [[] for _ in range(0)]  # will resize

    # We need to be careful: we build the graph as we go
    # Let's use a list-based approach
    vertex_list = [start_val]
    vertex_index = {start_val: 0}
    edge_list = []
    # For tracking adjacency to avoid duplicate edges
    # Since it's bipartite, we only add edges from point side to line side

    # Precompute all neighbors for a given vertex for speed
    def get_neighbors(val):
        side, coords = decode(val)
        neigh = []
        for first_val in range(q):
            y = [0] * d
            y[0] = first_val
            for c_idx in range(1, d):
                c = c_idx + 2
                a = rule_a[c] - 1
                b = rule_b[c] - 1
                if side == 0:
                    y[c_idx] = fadd(coords[c_idx], fmul(y[a], coords[b]))
                else:
                    y[c_idx] = fsub(coords[c_idx], fmul(coords[a], y[b]))
            neigh.append(encode(1 - side, tuple(y)))
        return neigh

    # BFS
    queue = deque([0])
    visited = [False] * 1  # dynamic
    visited[0] = True
    # Actually let's just iterate in order and extend
    for vi in range(len(vertex_list)):
        if time.time() > DEADLINE:
            break
        v = vertex_list[vi]
        side, _ = decode(v)
        for nb in get_neighbors(v):
            if nb not in vertex_index:
                vertex_index[nb] = len(vertex_list)
                vertex_list.append(nb)
            if side == 0:
                edge_list.append((vi, vertex_index[nb]))

    n = len(vertex_list)
    return n, edge_list
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize there's a potential issue. Let me reconsider the time budget. The BFS on a 4-regular graph with potentially thousands of vertices should be fast. But the `get_neighbors` function decodes and re-encodes, which involves division and modulo operations. Let me optimize further by keeping decoded coordinates in the queue.

Also, I should double-check: for k=4, g=10, d=5. The connected component size - let me think about what's expected. The LW graph CD(5,4) has 2*4^6 = 8192 total vertices, but the connected component of the zero point should be much smaller. Actually, for the LW construction, the graph is connected, so we'd get all 8192 vertices. That's way too many!

Wait, that can't be right. Let me re-examine. The LW graph CD(d,q) is defined on q^(d+1) points and q^(d+1) lines, so 2*q^(d+1) total vertices. For d=5, q=4: 2*4^6 = 8192. If it's connected, we get all of them. But the Moore bound for n(4,10) is 242, and the best known is 320. So 8192 is way too large.

Hmm, but the task says "the connected component of the zero point is returned." If the graph is connected, that's all vertices. Let me check if CD(5,4) is actually connected...

Actually, I think I'm misunderstanding the LW construction. Let me re-read the original code more carefully.

In the original code, `start = (0, (None,) + (0,) * d)`. So the point is (None, 0, 0, 0, 0, 0) for d=5. The coordinates are indexed 0..d where index 0 is None and indices 1..d are the actual coordinates.

The neighbor computation: for a point, y = [None, first] + [0]*(d-1), then for c=2..d, compute y[c]. So the line has coordinates y[1] through y[d], where y[1]=first.

For the zero point (all coords 0), the neighbors are lines with y[1]=first, and for c=2..d: y[c] = 0 + y[a]*0 = 0. So all neighbors of the zero point have all coordinates 0 except y[1]=first. That's 4 neighbors: lines (0,0,0,0), (1,0,0,0), (2,0,0,0), (3,0,0,0) in (y1,y2,y3,y4,y5).

This seems like it could produce a relatively small connected component. Let me just trust the original construction and focus on making it faster and adding the deadline guard.

Let me simplify my approach - keep the original logic but optimize the data structures:

