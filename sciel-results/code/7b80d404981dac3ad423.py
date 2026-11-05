import math
import time
from collections import deque

_DEADLINE = 130.0
_T0 = None


def _elapsed():
    return time.time() - _T0 if _T0 is not None else 0.0


def _time_left():
    return max(0.0, _DEADLINE - _elapsed())


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    A q-regular bipartite graph of girth >= d + 5 (for odd d).
    Returns (n, edges)."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

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
        c0 = c - r
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


def _verify_girth(n, edges, g):
    """Certify that the graph has no cycle shorter than g by enumerating
    non-backtracking walks from every vertex. Returns True if girth >= g."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    for s in range(n):
        visited = set()
        queue = deque()
        for w in adj[s]:
            state = (w, s)
            if state not in visited:
                visited.add(state)
                queue.append((w, s, 1))

        while queue:
            u, p, d = queue.popleft()
            if d >= g - 1:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s and d + 1 >= 2:
                    if d + 1 < g:
                        return False
                    continue
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, d + 1))

    return True


def _check_girth_cyclic_fast(m, a, b, g):
    """Fast check: BFS on non-backtracking state space of Z_m Cayley graph."""
    S = [a, -a, b, -b]
    inv_idx = [1, 0, 3, 2]

    visited = set()
    queue = [(0, -1, 0)]
    visited.add((0, -1))

    while queue:
        new_queue = []
        for (elem, prev, depth) in queue:
            if depth >= g - 1:
                continue
            for i in range(4):
                if prev >= 0 and i == inv_idx[prev]:
                    continue
                new_elem = (elem + S[i]) % m
                new_state = (new_elem, i)
                if new_state in visited:
                    continue
                if new_elem == 0 and depth + 1 >= 2:
                    if depth + 1 < g:
                        return False
                visited.add(new_state)
                new_queue.append((new_elem, i, depth + 1))
        queue = new_queue

    return True


def _check_girth_cyclic_array(m, a, b, g):
    """Array-based BFS on non-backtracking states. Returns True if girth >= g."""
    # state id: (elem * 4 + dir), dir in 0..3; also a special "start" state
    # We'll use dist[e][d] = shortest non-backtracking walk length from (0, -1) to (e, d)
    # where d is the last generator index used (or -1 for start).
    # We treat start as (0, -1) with dist 0.
    INF = 10**9
    dist = [[INF] * 4 for _ in range(m)]
    # Start: at element 0, no previous direction
    # We do a BFS where we track (elem, prev_dir) but start has prev_dir = -1.
    # Use a separate queue with (elem, prev_dir, depth). prev_dir in -1..3.
    # To use arrays, encode prev_dir: 0..3 for generators, 4 for start(-1).
    # Actually simpler: two-layer BFS.
    # dist[e][d] = min depth to reach (e, d) where d is last gen used.
    # From start (0, -1, 0), we can go to (a, 0, 1), (-a, 1, 1), (b, 2, 1), (-b, 3, 1).
    S = [a, (-a) % m, b, (-b) % m]
    inv = [1, 0, 3, 2]

    # Initialize
    for i in range(4):
        e = S[i]
        dist[e][i] = 1

    queue = deque()
    for i in range(4):
        queue.append((S[i], i))

    while queue:
        e, d = queue.popleft()
        cur_d = dist[e][d]
        if cur_d >= g - 1:
            continue
        for nd in range(4):
            if nd == inv[d]:
                continue
            ne = (e + S[nd]) % m
            nd_d = cur_d + 1
            if nd_d < dist[ne][nd]:
                dist[ne][nd] = nd_d
                # Check if we've returned to 0 with a cycle shorter than g
                # A cycle is formed when we reach element 0 via a non-backtracking walk of length >= 2
                if ne == 0 and nd_d >= 2:
                    if nd_d < g:
                        return False
                if nd_d < g - 1:
                    queue.append((ne, nd))

    return True


def _construct_fallback(k, g):
    return _luw_graph(k, g)


def construct(k, g):
    global _T0
    _T0 = time.time()

    if k != 4:
        return _construct_fallback(k, g)

    # Build the LUW graph (known valid, n=320 for k=4, g=10).
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Time-boxed cyclic-Cayley sweep
    if _time_left() > 5.0:
        moore_bound = 243
        max_m = min(n_luw - 1, 400)

        # Sweep m from moore_bound upward; for each m, try (a,b) pairs.
        # Use the array-based check which is faster for small m.
        for m in range(moore_bound, max_m + 1):
            if _time_left() < 3.0:
                break
            # Generate candidate (a, b) pairs with basic pruning
            # a in [1, m//2], b in [a+1, m//2]
            # Skip if 2a == m or 2b == m (order-2 elements)
            # Skip if a+b == m or a == m-b etc. (redundant)
            max_a = m // 2
            found = False
            for a in range(1, max_a + 1):
                if _time_left() < 2.0:
                    break
                if 2 * a == m:
                    continue
                # b > a, b <= m//2
                max_b = m // 2
                for b in range(a + 1, max_b + 1):
                    if _time_left() < 1.0:
                        break
                    if 2 * b == m:
                        continue
                    # Redundancy: a+b == m means b = m-a, but b <= m//2 and a >= 1
                    # so m-a >= m//2+1 > m//2, impossible. Skip check.
                    # a == b already excluded by b > a.
                    if math.gcd(math.gcd(a, b), m) != 1:
                        continue

                    if _check_girth_cyclic_array(m, a, b, g):
                        edge_set = set()
                        for i in range(m):
                            for s in [a, b]:
                                j = (i + s) % m
                                if i < j:
                                    edge_set.add((i, j))
                                else:
                                    edge_set.add((j, i))
                        edges = sorted(edge_set)
                        if m < best_n:
                            best_n = m
                            best_edges = edges
                        found = True
                        break
                if found:
                    break
            if found:
                break

    return best_n, best_edges
