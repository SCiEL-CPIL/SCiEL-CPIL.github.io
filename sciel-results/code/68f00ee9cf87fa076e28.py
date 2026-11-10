"""4-regular graph of girth 10: search circulants C_n(a,b), verify by
non-backtracking BFS (vertex-transitive => check from vertex 0), with a
guaranteed-valid Lazebnik-Ustimenko-Woldar CD(4,4) fallback."""
import time
import collections

_DEADLINE = 140.0
_TIME_LIMIT = _DEADLINE - 15.0
_START = None


def _now():
    return time.perf_counter() - (_START or 0)


def _adjacency(n, a, b):
    """Adjacency list for C_n(a,b); None if not a simple 4-regular graph."""
    a %= n
    b %= n
    if a == 0 or b == 0 or a == b or (a + b) % n == 0 or (2 * a) % n == 0 or (2 * b) % n == 0:
        return None
    adj = []
    for i in range(n):
        adj.append([(i + a) % n, (i - a) % n, (i + b) % n, (i - b) % n])
    return adj


def _girth_ok(adj, g):
    """True iff no non-backtracking closed walk of length < g through vertex 0.
    For vertex-transitive circulants, girth = min over all starts, so this suffices."""
    dq = collections.deque()
    dist = {}
    for nb in adj[0]:
        dq.append((nb, 0))
        dist[(nb, 0)] = 1
    while dq:
        cur, prev = dq.popleft()
        d = dist[(cur, prev)]
        if d >= g:
            break
        for nb in adj[cur]:
            if nb == prev:
                continue
            if nb == 0:
                if d + 1 < g:
                    return False
                continue
            st = (nb, cur)
            if st not in dist:
                dist[st] = d + 1
                dq.append(st)
    return True


def _edges_from_adj(adj):
    n = len(adj)
    seen = set()
    edges = []
    for u in range(n):
        for v in adj[u]:
            key = (u, v) if u < v else (v, u)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return edges


def _luw(k, g):
    """Seed Lazebnik-Ustimenko-Woldar CD(k, q=k) construction (guaranteed valid)."""
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


def construct(k, g):
    global _START
    _START = time.perf_counter()
    assert k == 4 and g == 10, "tuned for the (4,10) cage"

    # Moore bound for (4,10): 2 + 4*3^4 = 326. Known best: 320.
    # Search n from 320 down to 244 (lower bound stated in task).
    # For each n, try step pairs (a,b) with 1 <= a < b <= n//2, a+b != n,
    # 2a != n, 2b != n (simple 4-regular conditions).
    # Heuristic: try (1, b) first for various b, then other a.

    best_n = None
    best_edges = None

    for n in range(320, 243, -1):
        if _now() > _TIME_LIMIT:
            break
        # Candidate step pairs: a from 1..n//3, b from a+1..n//2
        # with constraints. Try (1,b) first (most structured), then others.
        # To limit work per n, cap the number of (a,b) pairs tried.
        pairs_tried = 0
        max_pairs = 200  # per n, to stay within budget

        # Phase 1: a=1, b from n//2 down to 2 (larger b = sparser = longer girth likely)
        for b in range(n // 2, 1, -1):
            if _now() > _TIME_LIMIT:
                return (best_n, best_edges) if best_n else _luw(k, g)
            if (1 + b) % n == 0 or (2 * 1) % n == 0:
                continue
            adj = _adjacency(n, 1, b)
            if adj is None:
                continue
            if _girth_ok(adj, g):
                return n, _edges_from_adj(adj)
            pairs_tried += 1
            if pairs_tried >= max_pairs:
                break

        # Phase 2: other a values (small a), various b
        if _now() <= _TIME_LIMIT:
            for a in range(2, min(n // 3, 15) + 1):
                if _now() > _TIME_LIMIT:
                    return (best_n, best_edges) if best_n else _luw(k, g)
                if (2 * a) % n == 0:
                    continue
                for b in range(n // 2, a + 1, -1):
                    if _now() > _TIME_LIMIT:
                        return (best_n, best_edges) if best_n else _luw(k, g)
                    if (a + b) % n == 0 or (2 * b) % n == 0:
                        continue
                    if abs(a - b) == 0 or abs(a - (n - b)) == 0:
                        continue
                    adj = _adjacency(n, a, b)
                    if adj is None:
                        continue
                    if _girth_ok(adj, g):
                        return n, _edges_from_adj(adj)
                    pairs_tried += 1
                    if pairs_tried >= max_pairs * 2:
                        break

    # Fallback: guaranteed-valid LUW construction
    return _luw(k, g)
