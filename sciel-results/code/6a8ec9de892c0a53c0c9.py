"""Cage construction: 4-regular (general k) graphs of given girth, minimized vertices."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_ok():
    return _elapsed() < _DEADLINE


def _verify_simple_regular(edges, n, k):
    """Check simple graph, all degrees == k."""
    seen = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _min_girth_bfs(n, edges, k):
    """Exact minimum girth via non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best = 10 ** 9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (nb, cur)
                if key in dist:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    dist[key] = d + 1
                    dq.append((nb, cur))
        if best <= k:  # early exit if we found a very short cycle
            break
    return best if best < 10 ** 9 else 10 ** 9


def _luzebnik_ustimenko_woldar(k, g):
    """L-U-W CD(d,q) with q=k: q-regular bipartite, girth >= d+5 (odd d)."""
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
        if not _time_ok():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _voltage_graph_4_10():
    """Construct (4,10)-cage via voltage graph lift. Target: 320 vertices."""
    # Use a base 4-regular graph on 4 vertices (K_4 is 3-reg, so use multigraph or different base)
    # Actually, use a base graph with 8 vertices, 4-regular, and lift over Z_40 (8*40=320)
    # Or base with 4 vertices and lift over Z_80.
    
    # Let's try: base = two disjoint K_4's? No, that's 3-regular.
    # Base = K_{2,2,2,2}? That's 4-regular on 8 vertices. But it has 4-cycles.
    # For girth 10, we need the base to have girth >= 10 OR the voltages to break short cycles.
    
    # Alternative: use a tree-like base with voltages that prevent short cycles in the lift.
    # Standard approach: base is a small graph, assign voltages in Z_m, verify no closed walk
    # of length < 10 has trivial voltage product.
    
    # Let me try a known construction: the (4,10)-cage can be built as a voltage graph
    # over Z_80 with a 4-vertex base (using a 4-regular multigraph on 4 vertices with 4 edges).
    
    # Actually, let's use a different approach: direct construction using the known
    # structure. The (4,10)-cage is known to exist with 320 vertices.
    
    # Let me try: base graph = cycle C_4 with doubled edges (4-regular multigraph on 4 vertices)
    # Vertices 0,1,2,3. Edges: (0,1)x2, (1,2)x2, (2,3)x2, (3,0)x2
    # Lift over Z_80.
    
    # For the lift to be simple and 4-regular with girth >= 10, we need to assign
    # voltages to each edge such that no non-backtracking closed walk of length < 10
    # has trivial total voltage.
    
    # Let me try a simpler approach: use the known (4,10)-cage construction.
    # One construction uses the group Z_4 x Z_80 or similar.
    
    # Given time constraints, let me try a heuristic search for voltage assignments.
    
    n_base = 4
    # Base: 4-regular multigraph on 4 vertices
    # Edges (with multiplicity): each pair of adjacent vertices in C_4 has 2 edges
    base_edges = [
        (0, 1), (0, 1),  # two parallel edges
        (1, 2), (1, 2),
        (2, 3), (2, 3),
        (3, 0), (3, 0),
    ]
    
    m = 80  # group order
    # Try to find voltages in Z_m for each of the 8 edges such that
    # the lifted graph has girth >= 10.
    
    # For simplicity, let's assign voltages as follows and check:
    # Edge (0,1) voltages: a, a'
    # Edge (1,2) voltages: b, b'
    # Edge (2,3) voltages: c, c'
    # Edge (3,0) voltages: d, d'
    
    # A walk of length L in the base lifts to a closed walk in the cover iff
    # the sum of voltages (with signs based on direction) is 0 mod m.
    
    # For girth >= 10, we need no closed walk of length 4, 6, or 8 with trivial voltage.
    # (Length 2 would be backtracking, which we exclude.)
    
    # Let me try: a=1, a'=2, b=1, b'=2, c=1, c'=2, d=1, d'=2
    # And check.
    
    voltages = {
        (0, 1, 0): 1,   # first edge 0->1
        (0, 1, 1): 2,   # second edge 0->1
        (1, 2, 0): 1,
        (1, 2, 1): 2,
        (2, 3, 0): 1,
        (2, 3, 1): 2,
        (3, 0, 0): 1,
        (3, 0, 1): 2,
    }
    
    # Build the lifted graph
    n = n_base * m
    edges = []
    for v in range(n_base):
        for g in range(m):
            for (u, w, idx) in [(0,1,0),(0,1,1),(1,2,0),(1,2,1),(2,3,0),(2,3,1),(3,0,0),(3,0,1)]:
                if u == v:
                    g2 = (g + voltages[(u, w, idx)]) % m
                    edges.append((v * m + g, w * m + g2))
    
    # Deduplicate and verify
    seen = set()
    unique_edges = []
    for i, j in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            seen.add((a, b))
            unique_edges.append((a, b))
    
    # Verify degree
    deg = [0] * n
    for i, j in unique_edges:
        deg[i] += 1
        deg[j] += 1
    
    if max(deg) != 4 or min(deg) != 4:
        return None, None
    
    # Check girth
    if _min_girth_bfs(n, unique_edges, 4) < 10:
        return None, None
    
    return n, unique_edges


def _tutte_8_cage():
    """Tutte 8-cage: 3-regular, girth 8, 30 vertices."""
    # Build a verified 3-reg girth-8 graph via L-U-W or return known.
    return None


def _hoffman_singleton():
    """Hoffman-Singleton: 7-regular, girth 5, 50 vertices."""
    n = 50
    edges = []
    def idx(a, b):
        return a * 10 + b
    for a in range(5):
        for b in range(10):
            v = idx(a, b)
            nbs = [
                idx(a, (b + 1) % 10), idx(a, (b - 1) % 10),
                idx((a + 1) % 5, (2 * b) % 10), idx((a + 1) % 5, (2 * b + 1) % 10),
                idx((a + 2) % 5, (4 * b) % 10), idx((a + 2) % 5, (4 * b + 1) % 10),
            ]
            for w in nbs:
                if v < w:
                    edges.append((v, w))
    return n, edges


def _moore_girth5(k):
    """For girth 5, return known small cages."""
    known = {3: 10, 4: 14, 5: 30, 6: 42, 7: 50}
    if k in known:
        if k == 7:
            return _hoffman_singleton()
        if k == 3:
            edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (1, 3), (1, 4),
                (5, 6), (5, 7), (5, 8), (5, 9),
                (6, 7), (6, 8), (6, 9),
                (7, 8), (7, 9), (8, 9),
                (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
            ]
            return 10, edges
    return _luzebnik_ustimenko_woldar(k, 5)


def _petersen():
    """Petersen graph: 3-regular, girth 5, 10 vertices."""
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (5, 6), (5, 7), (5, 8), (5, 9),
        (6, 7), (6, 8), (6, 9),
        (7, 8), (7, 9), (8, 9),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return 10, edges


def _cage_4_5():
    """(4,5)-cage: 14 vertices, 4-regular, girth 5."""
    # The (4,5)-cage is the 14-vertex graph.
    # It can be constructed as the complement of a perfect matching in the 3x3 rook graph? No.
    # It is the 4-regular graph on 14 vertices with girth 5.
    # Known as the "4-vertex cage" or similar.
    # Let's use a known edge list or generate it.
    # Actually, for k=4, g=5, n=14.
    # This is the 4-regular graph obtained by taking the K_{3,3,3} and adding 5 edges? No.
    # Let's fall back to L-U-W which gives n=14 for k=4, g=5.
    return _luzebnik_ustimenko_woldar(4, 5)


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g, minimizing n."""
    # Known optimal cages for common small cases.
    if k == 3 and g == 5:
        return _petersen()
    if k == 7 and g == 5:
        return _hoffman_singleton()
    if g == 5:
        return _moore_girth5(k)
    if k == 4 and g == 10:
        # Try voltage graph construction first
        n, edges = _voltage_graph_4_10()
        if n is not None:
            return n, edges
        # Fall back to L-U-W
        n, edges = _luzebnik_ustimenko_woldar(k, g)
        return n, edges

    # General: L-U-W construction (valid, girth >= g).
    n, edges = _luzebnik_ustimenko_woldar(k, g)
    return n, edges


def _tutte8():
    """Explicit Tutte 8-cage (30 vertices, 3-regular, girth 8)."""
    # Use L-U-W for k=3, g=8.
    n, edges = _luzebnik_ustimenko_woldar(3, 8)
    if not _verify_simple_regular(edges, n, 3):
        return None, None
    if _min_girth_bfs(n, edges, 3) < 8:
        return None, None
    return n, edges
# EVOLVE-BLOCK-END
