"""
Construct a k-regular graph with girth at least g.
Strategy: Systematic voltage-lift search with diverse algebraic voltage patterns,
plus a tree-extension fallback. Focused on k=3, g=16.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140

    if k == 3 and g == 16:
        best_result = None
        best_n = float('inf')

        # Try multiple base graphs and lift sizes with diverse voltage patterns
        base_graphs = _get_base_graphs()

        for base_n, base_edges in base_graphs:
            if time.time() > deadline - 15:
                break

            base_adj = _build_adj(base_n, base_edges)
            base_girth = _compute_girth(base_n, base_adj)
            if base_girth < 4:
                continue

            # Try lift sizes from small to large
            for m in range(32, 200, 2):
                if time.time() > deadline - 10:
                    break
                n = base_n * m
                if n > 950:
                    break

                # Try algebraic voltage patterns
                for pattern_idx in range(12):
                    if time.time() > deadline - 5:
                        break
                    voltages = _algebraic_voltages(base_edges, m, pattern_idx)
                    lift_edges = _build_lift(base_n, base_edges, voltages, m)
                    if _verify_girth(n, lift_edges, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, lift_edges)

                # Try Thue-Morse based patterns
                for shift in range(6):
                    if time.time() > deadline - 5:
                        break
                    voltages = _thue_morse_voltages(base_edges, m, shift)
                    lift_edges = _build_lift(base_n, base_edges, voltages, m)
                    if _verify_girth(n, lift_edges, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, lift_edges)

                # Try some random attempts
                for attempt in range(50):
                    if time.time() > deadline - 5:
                        break
                    voltages = _random_voltages(base_edges, m, attempt + m * 100)
                    lift_edges = _build_lift(base_n, base_edges, voltages, m)
                    if _verify_girth(n, lift_edges, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, lift_edges)

        # If we found something, return it
        if best_result is not None:
            return best_result

        # Fallback: LUW construction
        return _luw_construct(3, 16)

    # General case: LUW
    return _luw_construct(k, g)


def _get_base_graphs():
    """Get several 3-regular base graphs with good girth."""
    graphs = []

    # Petersen graph (10 vertices, girth 5)
    petersen = [(0,1),(0,4),(0,5),(1,3),(1,6),(2,4),(2,7),(3,5),(3,8),(4,8),(5,7),(6,8),(6,9),(7,9),(8,9)]
    # Wait, that's not right. Let me use the standard Petersen.
    petersen = [(0,1),(1,2),(2,0),(0,4),(1,5),(2,6),(3,4),(4,5),(5,6),(6,3)]
    # Actually let me just generate them properly
    graphs.append((10, _petersen_edges()))

    # 3-regular on 12 vertices
    for seed in [42, 137, 256, 512]:
        edges = _random_3reg(12, seed)
        if edges is not None:
            adj = _build_adj(12, edges)
            if _compute_girth(12, adj) >= 5:
                graphs.append((12, edges))

    # 3-regular on 14 vertices
    for seed in [77, 200, 333]:
        edges = _random_3reg(14, seed)
        if edges is not None:
            adj = _build_adj(14, edges)
            if _compute_girth(14, adj) >= 5:
                graphs.append((14, edges))

    # 3-regular on 16 vertices
    for seed in [11, 99, 222]:
        edges = _random_3reg(16, seed)
        if edges is not None:
            adj = _build_adj(16, edges)
            if _compute_girth(16, adj) >= 5:
                graphs.append((16, edges))

    # Deduplicate
    seen = set()
    unique = []
    for n, edges in graphs:
        key = frozenset(frozenset(e) for e in edges)
        if key not in seen:
            seen.add(key)
            unique.append((n, edges))
    return unique


def _petersen_edges():
    """Standard Petersen graph edges."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
             (5,7),(7,9),(9,6),(6,8),(8,5),
             (0,5),(1,6),(2,7),(3,8),(4,9)]
    return edges


def _random_3reg(n, seed):
    """Generate a random 3-regular graph on n vertices (n must be even)."""
    if n % 2 != 0 or n < 4:
        return None

    random.seed(seed)

    for _ in range(500):
        stubs = list(range(n * 3))
        random.shuffle(stubs)

        edges = []
        edge_set = set()
        valid = True

        for i in range(0, n * 3, 2):
            u = stubs[i] // 3
            v = stubs[i + 1] // 3
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edges.append((u, v))
            edge_set.add(key)

        if valid and len(edges) == n * 3 // 2:
            # Verify all degrees are 3
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == 3 for d in deg):
                return edges

    return None


def _build_adj(n, edges):
    """Build adjacency list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _compute_girth(n, adj):
    """Compute the girth of a graph using BFS from each vertex."""
    min_girth = float('inf')

    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len

        if min_girth == 3:
            break

    return min_girth if min_girth != float('inf') else 0


def _algebraic_voltages(base_edges, m, pattern_idx):
    """Generate algebraic voltage assignments based on modular arithmetic."""
    voltages = {}
    num_edges = len(base_edges)

    for idx, (u, v) in enumerate(base_edges):
        if pattern_idx == 0:
            w = (idx * 7 + 3) % m
        elif pattern_idx == 1:
            w = (idx * idx + idx + 1) % m
        elif pattern_idx == 2:
            w = (idx * 13 + 5) % m
        elif pattern_idx == 3:
            w = (idx * idx * idx + idx * 2 + 1) % m
        elif pattern_idx == 4:
            w = (idx * 11 + 7) % m
        elif pattern_idx == 5:
            w = (idx * idx * 3 + idx * 5 + 2) % m
        elif pattern_idx == 6:
            w = (idx * 19 + 11) % m
        elif pattern_idx == 7:
            w = (idx * idx * idx + idx * 3 + 7) % m
        elif pattern_idx == 8:
            w = (idx * 23 + 13) % m
        elif pattern_idx == 9:
            w = (idx * idx * 7 + idx * 11 + 3) % m
        elif pattern_idx == 10:
            w = (idx * 31 + 17) % m
        else:
            w = (idx * idx * 11 + idx * 13 + 5) % m

        voltages[(u, v)] = w
        voltages[(v, u)] = (m - w) % m

    return voltages


def _thue_morse_voltages(base_edges, m, shift):
    """Generate Thue-Morse based voltage assignments."""
    voltages = {}
    num_edges = len(base_edges)

    for idx, (u, v) in enumerate(base_edges):
        # Thue-Morse: parity of number of 1s in binary representation
        tm_val = bin(idx + shift).count('1') % 2
        # Combine with modular arithmetic for diversity
        w = (tm_val * (m // 2) + idx * 3 + shift) % m
        voltages[(u, v)] = w
        voltages[(v, u)] = (m - w) % m

    return voltages


def _random_voltages(base_edges, m, seed):
    """Generate random voltage assignments."""
    random.seed(seed)
    voltages = {}
    for u, v in base_edges:
        w = random.randint(0, m - 1)
        voltages[(u, v)] = w
        voltages[(v, u)] = (m - w) % m
    return voltages


def _build_lift(base_n, base_edges, voltages, m):
    """
    Build the voltage lift of a base graph.
    Vertex (u, i) in lift corresponds to vertex u in base at sheet i.
    Edge (u,v) with voltage w connects (u,i) to (v, (i+w)%m).
    """
    n = base_n * m
    lift_edges = []

    for u, v in base_edges:
        w = voltages[(u, v)]
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + ((i + w) % m)
            if u_lift < v_lift:
                lift_edges.append((u_lift, v_lift))
            else:
                lift_edges.append((v_lift, u_lift))

    return lift_edges


def _verify_girth(n, edges, g):
    """
    Verify the graph is simple, 3-regular, and has no cycle shorter than g.
    Uses BFS from each vertex to find shortest cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)

    # Check simplicity and degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
        if len(set(adj[i])) != 3:
            return False

    # Check no short edges (multi-edges)
    for i in range(n):
        for j in adj[i]:
            if i in adj[j] and adj[j].count(i) > 1:
                return False

    # BFS from each vertex to check girth
    max_depth = g // 2

    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])

        while queue:
            u = queue.popleft()
            depth = dist[u]

            if depth >= max_depth:
                continue

            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = depth + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[u] != v and parent[v] != u:
                        cycle_len = depth + 1 + dist[v]
                        if cycle_len < g:
                            return False

    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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

# EVOLVE-BLOCK-END
