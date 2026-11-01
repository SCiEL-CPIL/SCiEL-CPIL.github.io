"""
Construct a k-regular graph with girth at least g.
Voltage lift with Thue-Morse and random strategies for k=3, g=16.
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

        # Strategy 1: Voltage lift with various base graphs and lift sizes
        for base_n in [6, 8, 10, 12]:
            if time.time() > deadline - 10:
                break

            base_edges = _random_3reg(base_n, base_n * 7 + 13)
            if base_edges is None:
                continue

            base_adj = _build_adj(base_n, base_edges)
            base_girth = _compute_girth(base_n, base_adj)
            if base_girth < 4:
                continue

            for m in range(40, 120, 2):
                if time.time() > deadline - 5:
                    break
                n = base_n * m
                if n > 900:
                    break

                # Thue-Morse based voltage assignment
                result = _try_thue_morse_lift(base_n, base_edges, m, g, deadline)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)

                # Random voltage assignments
                for attempt in range(200):
                    if time.time() > deadline - 5:
                        break
                    result = _try_random_lift(base_n, base_edges, m, g, deadline)
                    if result is not None:
                        n_r, edges_r = result
                        if n_r < best_n:
                            best_n = n_r
                            best_result = (n_r, edges_r)

        if best_result is not None:
            return best_result

        # Strategy 2: Try larger lift sizes with fewer attempts
        for base_n in [6, 8]:
            if time.time() > deadline - 5:
                break
            base_edges = _random_3reg(base_n, base_n * 11 + 7)
            if base_edges is None:
                continue
            base_adj = _build_adj(base_n, base_edges)
            if _compute_girth(base_n, base_adj) < 4:
                continue

            for m in range(120, 200, 4):
                if time.time() > deadline - 3:
                    break
                n = base_n * m
                if n > 1000:
                    break
                for attempt in range(100):
                    if time.time() > deadline - 3:
                        break
                    result = _try_random_lift(base_n, base_edges, m, g, deadline)
                    if result is not None:
                        n_r, edges_r = result
                        if n_r < best_n:
                            best_n = n_r
                            best_result = (n_r, edges_r)

        if best_result is not None:
            return best_result

        # Fallback: LUW
        return _luw_construct(3, 16)

    # General case: LUW
    return _luw_construct(k, g)


def _random_3reg(n, seed):
    """Generate a random 3-regular graph on n vertices (n must be even)."""
    if n % 2 != 0 or n < 4:
        return None

    random.seed(seed)

    for _ in range(200):
        stubs = list(range(n * 3))
        random.shuffle(stubs)

        edges = []
        edge_set = set()
        degree = [0] * n
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
            degree[u] += 1
            degree[v] += 1

        if valid and all(d == 3 for d in degree):
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


def _thue_morse_sequence(length):
    """Generate Thue-Morse sequence of given length."""
    return [bin(i).count('1') % 2 for i in range(length)]


def _try_thue_morse_lift(base_n, base_edges, m, g, deadline):
    """Try a voltage lift using Thue-Morse based voltage assignments."""
    tm = _thue_morse_sequence(len(base_edges) * 4)

    for shift in range(4):
        voltages = {}
        for idx, (u, v) in enumerate(base_edges):
            w = tm[idx + shift] * (m // 2) + (idx * 7) % m
            voltages[(u, v)] = w % m
            voltages[(v, u)] = (m - w) % m

        lift_edges = _build_lift(base_n, base_edges, voltages, m)

        if _check_girth_fast(base_n * m, lift_edges, g):
            return base_n * m, lift_edges

    return None


def _try_random_lift(base_n, base_edges, m, g, deadline):
    """Try a random voltage lift."""
    voltages = {}
    for u, v in base_edges:
        w = random.randint(0, m - 1)
        voltages[(u, v)] = w
        voltages[(v, u)] = (m - w) % m

    lift_edges = _build_lift(base_n, base_edges, voltages, m)
    n = base_n * m

    if _check_girth_fast(n, lift_edges, g):
        return n, lift_edges

    return None


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


def _check_girth_fast(n, edges, g):
    """
    Fast girth check: verify all vertices have degree 3 and no cycle shorter than g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

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
