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

        for m in range(moore_bound, max_m + 1):
            if _time_left() < 3.0:
                break
            max_a = min(m // 2, 50)
            for a in range(1, max_a + 1):
                if _time_left() < 2.0:
                    break
                if 2 * a == m:
                    continue
                max_b = min(m // 2, a + 50)
                for b in range(a + 1, max_b + 1):
                    if _time_left() < 1.0:
                        break
                    if 2 * b == m:
                        continue
                    if a == b or a == m - b or b == m - a:
                        continue
                    if math.gcd(math.gcd(a, b), m) != 1:
                        continue

                    if _check_girth_cyclic_fast(m, a, b, g):
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
                        break
                if best_n == m:
                    break
            if best_n == m:
                break

    return best_n, best_edges
