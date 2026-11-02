"""Rewritten: randomized local-search for small girth-10 4-regular graphs,
with the Lazebnik-Ustimenko-Woldar graph as a guaranteed-valid fallback."""
import time
import random
import math

_DEADLINE = 150.0          # evaluator wall-clock limit
_SAFETY = 10.0             # safety margin before the limit
_START = None

def _now():
    return time.time()

def _time_left():
    if _START is None:
        return _DEADLINE
    return _DEADLINE - _SAFETY - (_now() - _START)


# ---------------------------------------------------------------------------
# Girth certification via non-backtracking walks from every vertex.
# Returns the girth (minimum cycle length). A graph with no cycle < g has
# girth >= g (or is a tree, but a 4-regular graph always has cycles).
# ---------------------------------------------------------------------------
def _girth(adj, n):
    """adj: list of sorted neighbor lists. Returns girth (int)."""
    best = 10 ** 9
    for s in range(n):
        # non-backtracking BFS: state (v, prev)
        from collections import deque
        seen = {}
        q = deque()
        for nb in adj[s]:
            seen[(s, nb)] = 1
            q.append((s, nb, 1))
        while q:
            prev, v, dist = q.popleft()
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s and dist + 1 >= 3:
                    cyc = dist + 1
                    if cyc < best:
                        best = cyc
                    # do not expand further from a return to s
                    continue
                key = (v, nb)
                if key in seen:
                    cyc = seen[key] + dist + 1
                    if cyc < best:
                        best = cyc
                    continue
                seen[key] = dist + 1
                q.append((v, nb, dist + 1))
    return best


# ---------------------------------------------------------------------------
# Random 4-regular simple graph via configuration model (with dedup).
# ---------------------------------------------------------------------------
def _random_k_regular(n, k, rng):
    if n * k % 2:
        return None
    stubs = [v for v in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            return None
        if b in adj[a]:
            return None
        adj[a].add(b)
        adj[b].add(a)
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return [sorted(a) for a in adj]


def _edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# 2-switch move: pick two edges (a,b),(c,d) with all distinct endpoints and
# no existing cross edges, replace with (a,c),(b,d) (or a variant).
# ---------------------------------------------------------------------------
def _try_2switch(adj, rng):
    n = len(adj)
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    if len(edge_list) < 2:
        return False
    for _ in range(60):
        i = rng.randrange(len(edge_list))
        j = rng.randrange(len(edge_list))
        if i == j:
            continue
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            continue
        # try pairing (a,c),(b,d)
        if (c in adj[a] or d in adj[b] or b in adj[c] or a in adj[d]):
            continue
        # perform switch
        adj[a].remove(b); adj[b].remove(a)
        adj[a].append(c); adj[b].append(d)
        adj[c].append(a); adj[d].append(b)
        adj[a].sort(); adj[b].sort(); adj[c].sort(); adj[d].sort()
        return True
    return False


# ---------------------------------------------------------------------------
# Local search: raise the girth.
# ---------------------------------------------------------------------------
def _local_search(n, k, g, rng, time_budget):
    t0 = _now()
    adj = _random_k_regular(n, k, rng)
    if adj is None:
        return None
    cur_g = _girth(adj, n)
    best_adj = [list(a) for a in adj]
    best_g = cur_g
    iters = 0
    while _now() - t0 < time_budget and _time_left() > 0:
        iters += 1
        if cur_g >= g:
            # found a valid graph; keep refining for smaller? we fixed n, so done
            best_adj = [list(a) for a in adj]
            best_g = cur_g
            break
        if _try_2switch(adj, rng):
            new_g = _girth(adj, n)
            if new_g >= cur_g:
                cur_g = new_g
                if cur_g > best_g:
                    best_g = cur_g
                    best_adj = [list(a) for a in adj]
            else:
                # revert: we don't store old edges; simplest is to restart
                # if we can't cheaply revert, just accept a fresh graph
                adj = _random_k_regular(n, k, rng)
                if adj is None:
                    break
                cur_g = _girth(adj, n)
    if best_g >= g:
        return best_adj
    return None


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar fallback (valid, large).
# ---------------------------------------------------------------------------
def _luw(k, g):
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Moore bound for (k, g) with g even: 1 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
# ---------------------------------------------------------------------------
def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        return 1 + k * sum((k - 1) ** i for i in range(m - 1))
    else:
        m = (g - 1) // 2
        return k * sum((k - 1) ** i for i in range(m))


def construct(k, g):
    global _START
    _START = _now()
    rng = random.Random(12345)

    moore = _moore_bound(k, g)
    # Candidate n values to try, near the known optimum (320 for (4,10)) and
    # between the Moore bound and a small upper cap.
    if k == 4 and g == 10:
        candidates = [320, 300, 280, 260, 243]
    else:
        candidates = []
        # try a few multiples around the Moore bound
        base = max(moore, k * (k - 1) + 1)
        for factor in (1.0, 1.1, 1.25, 1.5, 1.8):
            candidates.append(int(base * factor))
        candidates = sorted(set(candidates))

    best_result = None
    for n in candidates:
        if _time_left() < 20:
            break
        # give this n a share of the remaining budget
        budget = min(30.0, max(5.0, _time_left() / 2.0))
        adj = _local_search(n, k, g, rng, budget)
        if adj is not None:
            # verify girth exactly
            if _girth(adj, n) >= g:
                edges = _edges_from_adj(adj)
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                # found a valid graph at this n; since candidates are sorted
                # ascending in the (4,10) case, smaller n is better, but we
                # iterate from larger to smaller; keep the smallest valid.
    if best_result is not None:
        return best_result

    # Fallback: LUW graph (valid, large)
    return _luw(k, g)
