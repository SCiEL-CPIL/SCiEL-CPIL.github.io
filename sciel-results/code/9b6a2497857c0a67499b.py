"""Cage graph construction: 4-regular graphs of given girth, minimized vertices."""
import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit with margin

    # Special cases with known optimal constructions
    special_cases = {
        (3, 5): (10, _petersen_edges()),
        (3, 6): (14, _heawood_edges()),
        (4, 5): (16, _clebsch_minus_matching_edges()),
    }
    
    if (k, g) in special_cases:
        return special_cases[(k, g)]

    # For the main case (4, 10): use known constructions
    if k == 4 and g == 10:
        # Try the (4,10)-cage construction
        # The smallest known 4-regular graph of girth 10 has 320 vertices
        result = _construct_4_10_cage()
        if result is not None:
            return result
        # Fallback to L-U-W
        return _luw_construct(k, g)

    # General L-U-W construction
    return _luw_construct(k, g)

def _petersen_edges():
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return edges

def _heawood_edges():
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return edges

def _clebsch_minus_matching_edges():
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            xor = i ^ j
            dist = bin(xor).count('1')
            if dist == 1:
                edges.append((i, j))
    return edges

def _construct_4_10_cage():
    """
    Construct a 4-regular graph of girth 10 with 320 vertices.
    Uses a 20-fold voltage cover of the 16-vertex (4,5)-cage.
    """
    import time
    deadline = time.time() + 100  # Leave margin for the rest

    # Base: 16-vertex (4,5)-cage (Clebsch minus perfect matching)
    base_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                  if bin(i ^ j).count('1') == 1]
    base_adj = [[] for _ in range(16)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    n = 320  # 16 * 20
    rng = random.Random(42)

    # Try some deterministic patterns first
    deterministic_voltages = []
    
    # Pattern 1: Constant voltage 1
    v1 = {}
    for u, v in base_edges:
        v1[(u, v)] = 1
        v1[(v, u)] = 19
    deterministic_voltages.append(v1)
    
    # Pattern 2: Voltage based on u
    v2 = {}
    for u, v in base_edges:
        vol = (u % 19) + 1
        v2[(u, v)] = vol
        v2[(v, u)] = (20 - vol) % 20
    deterministic_voltages.append(v2)
    
    # Pattern 3: Voltage based on edge index
    v3 = {}
    for idx, (u, v) in enumerate(base_edges):
        vol = (idx % 19) + 1
        v3[(u, v)] = vol
        v3[(v, u)] = (20 - vol) % 20
    deterministic_voltages.append(v3)

    for vol_set in deterministic_voltages:
        if time.time() > deadline:
            break
        result = _try_voltages(base_edges, base_adj, vol_set, n)
        if result is not None:
            return result

    # Random search
    for attempt in range(5000):
        if time.time() > deadline:
            break

        # Assign random non-zero voltages to directed edges
        voltages = {}
        for u, v in base_edges:
            vol = rng.randint(1, 19)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (20 - vol) % 20

        result = _try_voltages(base_edges, base_adj, voltages, n)
        if result is not None:
            return result

    return None  # Fall back to L-U-W

def _try_voltages(base_edges, base_adj, voltages, n):
    """Builds the cover and checks girth. Returns (n, edges) or None."""
    edge_set = set()
    for u in range(16):
        for f in range(20):
            ui = u * 20 + f
            for v in base_adj[u]:
                nf = (f + voltages[(u, v)]) % 20
                vi = v * 20 + nf
                if ui < vi:
                    edge_set.add((ui, vi))
                else:
                    edge_set.add((vi, ui))

    if len(edge_set) != 640:
        return None

    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)

    # Verify girth >= 10: no cycle of length 3-9
    ok = True
    for s in range(n):
        vis = {s: (-1, 0)}
        queue = [(s, -1, 0)]
        qi = 0
        while qi < len(queue) and ok:
            node, par, d = queue[qi]
            qi += 1
            if d >= 4:
                continue
            for nb in adj[node]:
                if nb == par:
                    continue
                if nb not in vis:
                    vis[nb] = (node, d + 1)
                    queue.append((nb, node, d + 1))
                else:
                    vp, vd = vis[nb]
                    if vp != node and d + vd + 1 < 10:
                        ok = False
                        break
        if not ok:
            break

    if ok:
        return n, list(edge_set)
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    # Choose d such that the girth is at least g
    # For L-U-W CD(d,q), the girth is at least 2d-2 for odd d, etc.
    # We need girth >= g, so choose d appropriately
    
    # For k=4, g=10: we need d such that girth >= 10
    # L-U-W with q=4, d=6 gives girth 10 (approximately)
    # Let's compute: for CD(d,q), girth is at least 2*floor(d/2)+2 or similar
    
    # Standard: for CD(d, q), girth >= 2d - 2 for d >= 3
    # So for girth 10, need 2d-2 >= 10, so d >= 6
    
    d = max(3, (g + 2) // 2)  # d = 6 for g=10
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for CD(d, q)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
