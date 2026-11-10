"""Search-based solver for small cage graphs n(k,g).

Strategy (combined):
  1) LUW CD(d,q) construction as a guaranteed valid baseline (fast, no search).
  2) Time-budgeted randomized search for a smaller k-regular graph of girth >= g:
       - generate random k-regular simple graphs (configuration model + swap repair),
       - refine via 2-switches that preserve k-regularity and never close a cycle
         shorter than g (fast local girth test),
       - certify the best candidate exactly (BFS-ball tree test).
  The exact girth test for even g: every radius r = g//2 - 1 BFS ball must be a
  tree with exactly 1 + k*((k-1)^r - 1)/(k-2) vertices, and no non-tree edge may
  close a cycle of length < g.
"""
import time
import random
from collections import deque

# ---- deadline guard ----
_DEADLINE = time.time() + 145.0   # 150 s wall clock, 5 s safety margin

def _deadline_reached():
    return time.time() >= _DEADLINE


# ---------------- LUW baseline ----------------
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) graph, q = k. Returns (n, edges)."""
    q = k
    d0 = g - 5
    if d0 < 1:
        d0 = 1
    if d0 % 2 == 0:
        d0 += 1
    d = d0

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


# ---------------- girth certification ----------------
def _tree_ball_size(k, r):
    """Number of vertices in a k-regular tree ball of radius r."""
    if r == 0:
        return 1
    if k == 2:
        return 1 + 2 * r
    return 1 + k * ((k - 1) ** r - 1) // (k - 2)


def _girth_ok(adj, n, k, g):
    """Exact: True iff graph has girth >= g.
    For even g: every radius r = g//2 - 1 BFS ball must be a tree of the exact
    expected size, and no non-tree edge may close a cycle of length < g.
    For odd g: use radius (g-1)//2 with a relaxed (>=) size test + cycle check.
    """
    if g <= 3:
        return True
    if g % 2 == 0:
        r = g // 2 - 1
        expected = _tree_ball_size(k, r)
        exact = True
    else:
        r = (g - 1) // 2
        expected = _tree_ball_size(k, r)
        exact = False

    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        count = 1
        while q:
            v = q.popleft()
            dv = dist[v]
            if dv >= r:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    count += 1
                    if dv + 1 <= r:
                        q.append(u)
                elif u != parent[v]:
                    # non-tree edge inside / near the ball
                    if dv + dist[u] + 1 < g:
                        return False
        if exact and count != expected:
            return False
        if (not exact) and count < expected:
            return False
    return True


def _local_girth_ok(adj, n, k, g, s):
    """Cheap necessary girth test from a single start vertex s."""
    if g <= 3:
        return True
    if g % 2 == 0:
        r = g // 2 - 1
        expected = _tree_ball_size(k, r)
    else:
        r = (g - 1) // 2
        expected = _tree_ball_size(k, r)
    dist = [-1] * n
    parent = [-1] * n
    dist[s] = 0
    q = deque([s])
    count = 1
    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= r:
            continue
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dv + 1
                parent[u] = v
                count += 1
                if dv + 1 <= r:
                    q.append(u)
            elif u != parent[v]:
                if dv + dist[u] + 1 < g:
                    return False
    if count < expected:
        return False
    return True


# ---------------- random k-regular graph generation ----------------
def _try_config(n, k, rng):
    """Configuration model; returns adjacency sets or None."""
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b or b in adj[a]:
            return None
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _repair_regular(adj, n, k, rng, iters=20000):
    """Use 2-switches to repair degree defects and remove multi-edges."""
    for _ in range(iters):
        bad = None
        for i in range(n):
            if len(adj[i]) != k:
                bad = i
                break
        if bad is None:
            # check multi-edges / loops
            multi = False
            for i in range(n):
                if len(adj[i]) != len(set(adj[i])):
                    multi = True
                    break
                if i in adj[i]:
                    multi = True
                    break
            if not multi:
                return True
            bad = None
        # find a vertex with wrong degree to fix
        for i in range(n):
            if len(adj[i]) != k:
                bad = i
                break
        if bad is None:
            continue
        # try a random 2-switch to change degrees
        i = rng.randrange(n)
        if len(adj[i]) != k:
            continue
        j_list = list(adj[i])
        if not j_list:
            continue
        j = rng.choice(j_list)
        a = rng.randrange(n)
        if a == i or a == j:
            continue
        b_list = list(adj[a])
        if not b_list:
            continue
        b = rng.choice(b_list)
        if b == i or b == j or b == a or b in adj[i] or j in adj[a] or a in adj[i] or i in adj[a]:
            continue
        # swap i-j, a-b -> i-b, a-j
        adj[i].remove(j)
        adj[j].remove(i)
        adj[a].remove(b)
        adj[b].remove(a)
        adj[i].add(b)
        adj[b].add(i)
        adj[a].add(j)
        adj[j].add(a)
    return False


def _random_regular_graph(n, k, rng):
    if n * k % 2 != 0 or n < k + 1:
        return None
    for _ in range(300):
        adj = _try_config(n, k, rng)
        if adj is not None and all(len(x) == k for x in adj):
            return adj
        if adj is not None:
            if _repair_regular(adj, n, k, rng, iters=5000):
                return adj
    return None


# ---------------- 2-switch refinement ----------------
def _refine(adj, n, k, g, rng, steps):
    """Apply random 2-switches that keep k-regularity and never close a cycle < g."""
    for _ in range(steps):
        i = rng.randrange(n)
        j_list = list(adj[i])
        if len(j_list) < 2:
            continue
        j = rng.choice(j_list)
        a = rng.randrange(n)
        if a == i or a == j:
            continue
        b_list = list(adj[a])
        if not b_list:
            continue
        b = rng.choice(b_list)
        if b == i or b == j or b == a or b in adj[i] or j in adj[a] or a in adj[i] or i in adj[a]:
            continue
        # candidate switch i-j, a-b -> i-b, a-j
        # check it does not create a short cycle: remove j from i and b from a,
        # add b to i and j to a, then local girth test around affected area.
        adj[i].remove(j)
        adj[j].remove(i)
        adj[a].remove(b)
        adj[b].remove(a)
        adj[i].add(b)
        adj[b].add(i)
        adj[a].add(j)
        adj[j].add(a)
        # cheap local check around the 4 affected vertices
        ok = True
        for s in (i, j, a, b):
            if not _local_girth_ok(adj, n, k, g, s):
                ok = False
                break
        if not ok:
            # revert
            adj[i].remove(b)
            adj[b].remove(i)
            adj[a].remove(j)
            adj[j].remove(a)
            adj[i].add(j)
            adj[j].add(i)
            adj[a].add(b)
            adj[b].add(a)
    return adj


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 145.0

    # Baseline: LUW
    try:
        n_luw, edges_luw = _luw_construct(k, g)
    except Exception:
        n_luw, edges_luw = None, None

    best_n = n_luw if n_luw is not None else float('inf')
    best_edges = edges_luw if edges_luw is not None else []

    if best_n is None or best_n == float('inf'):
        return 0, []

    # Moore bound lower bound
    if g >= 4:
        if g % 2 == 0:
            lb = 2 * (k - 1) * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else 2 * (k - 1) ** (g // 2)
        else:
            lb = (k - 1) * ((k - 1) ** ((g + 1) // 2) - 1) // (k - 2) + 1 if k > 2 else (k - 1) ** ((g + 1) // 2)
        lb = max(lb, 2 * k)
    else:
        lb = 2 * k

    if _deadline_reached():
        return best_n, best_edges

    # Randomized search for smaller n
    rng = random.Random(987654321)
    step = 2 if k % 2 == 1 else 1
    lo = max(lb, k + 1)
    if lo % step:
        lo += step
    hi = min(best_n - 1, lo + 300)  # search a window above the Moore bound

    if hi < lo:
        return best_n, best_edges

    n = lo
    while n <= hi and not _deadline_reached():
        if n < best_n:
            max_attempts = 6 if n < 700 else 3
            for _ in range(max_attempts):
                if _deadline_reached():
                    break
                adj = _random_regular_graph(n, k, rng)
                if adj is None:
                    continue
                # refine with 2-switches
                _refine(adj, n, k, g, rng, steps=400)
                # exact certification
                if _girth_ok(adj, n, k, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    best_n = n
                    best_edges = edges
                    return best_n, best_edges
        n += step

    return best_n, best_edges
