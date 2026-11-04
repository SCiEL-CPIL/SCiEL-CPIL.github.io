"""Deterministic voltage-graph (cyclic cover) construction of small 3-regular
graphs of girth >= g, with randomized restarts over base graphs and voltage
assignments.  The girth is certified exactly via non-backtracking BFS."""
import random
import sys
import time
from collections import deque


def _girth_nonbacktracking(n, adj):
    """Exact girth via non-backtracking BFS from every vertex.
    Returns length of shortest cycle (>=3) or a large number if acyclic."""
    best = 10**9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq.append((s, nb, 1))
        while dq:
            v, prev, d = dq.popleft()
            if d >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    cyc = d + 1
                    if cyc < best:
                        best = cyc
                    continue
                st = (v, nb)
                if st in dist:
                    cyc = d + dist[st]
                    if cyc < best:
                        best = cyc
                else:
                    dist[st] = d + 1
                    dq.append((v, nb, d + 1))
    return best


def _is_k_regular(adj, n, k):
    return all(len(adj[i]) == k for i in range(n))


def _build_lifting(base_adj, base_edges, m, voltages):
    """Build the cyclic voltage lift.
    base_adj: list of neighbor lists for base graph
    base_edges: list of (u, v) undirected edges in base
    m: covering modulus
    voltages: dict mapping edge (min(u,v), max(u,v)) -> integer in [0, m-1]
    Returns (n, adj) for the lifted graph, or None if invalid.
    """
    n_base = len(base_adj)
    n = n_base * m
    if n == 0:
        return None
    adj = [[] for _ in range(n)]

    for (u, v) in base_edges:
        key = (min(u, v), max(u, v))
        vol = voltages.get(key, 0)
        # Edge u->v has voltage vol: vertex (u, t) connects to (v, t+vol mod m)
        # Edge v->u has voltage -vol: vertex (v, t) connects to (u, t-vol mod m)
        for t in range(m):
            a = u * m + t
            b = v * m + ((t + vol) % m)
            adj[a].append(b)
            adj[b].append(a)

    # Verify 3-regular and simple
    if not _is_k_regular(adj, n, 3):
        return None
    # Check for multi-edges
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None
    return n, adj


def _random_base_tree3(k, g, rng, size=8):
    """Generate a random 3-regular base graph candidate by starting from a tree
    and adding edges to make it 3-regular, keeping girth high."""
    # Start with a random tree
    n = size
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        p = rng.randrange(i)
        adj[p].append(i)
        adj[i].append(p)

    # Now add edges to make it 3-regular
    # Each vertex needs 3 - len(adj[i]) more edges
    frontier = [v for v in range(n) if len(adj[v]) < 3]
    attempts = 0
    while frontier and attempts < 100:
        attempts += 1
        rng.shuffle(frontier)
        a = frontier[0]
        cand = [b for b in frontier[1:] if b not in adj[a] and b != a]
        if not cand:
            break
        rng.shuffle(cand)
        b = cand[0]
        adj[a].append(b)
        adj[b].append(a)
        frontier = [v for v in range(n) if len(adj[v]) < 3]

    if not _is_k_regular(adj, n, 3):
        return None

    # Build edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))

    # Check girth of base
    gr = _girth_nonbacktracking(n, adj)
    if gr < 5:  # base needs decent girth
        return None

    return adj, edges


def _try_voltage_lift(base_adj, base_edges, g, m, rng, time_limit=15):
    """Try random voltage assignments on the base graph to find a lift with
    girth >= g. Returns (n, edges) if found, else None."""
    start_time = time.time()
    n_base = len(base_adj)
    n_lift = n_base * m

    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break

        # Assign random voltages
        voltages = {}
        for (u, v) in base_edges:
            key = (min(u, v), max(u, v))
            voltages[key] = rng.randrange(m)

        n, adj = _build_lifting(base_adj, base_edges, m, voltages)
        if n is None:
            continue

        gr = _girth_nonbacktracking(n, adj)
        if gr >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    return None


def _try_direct_lift_with_optimization(base_adj, base_edges, g, m, rng, time_limit=15):
    """Try voltage assignments with some heuristic optimization:
    use voltages that spread out to avoid short cycles."""
    start_time = time.time()
    n_base = len(base_adj)
    n_lift = n_base * m

    # Collect all cycles in base up to length g
    # For each cycle, the total voltage must not be 0 mod m (for odd cycles)
    # or the lift creates a shorter cycle

    for attempt in range(80):
        if time.time() - start_time > time_limit:
            break

        voltages = {}
        for (u, v) in base_edges:
            key = (min(u, v), max(u, v))
            voltages[key] = rng.randrange(m)

        n, adj = _build_lifting(base_adj, base_edges, m, voltages)
        if n is None:
            continue

        gr = _girth_nonbacktracking(n, adj)
        if gr >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    return None


def _find_small_base(k, g, rng, max_size=16):
    """Find a small 3-regular base graph with girth >= 5."""
    for size in range(6, max_size + 1, 2):
        if size * k % 2 != 0:
            continue
        for attempt in range(20):
            adj = _random_k_regular(size, k, rng)
            if adj is None:
                continue
            edges = []
            for i in range(size):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            gr = _girth_nonbacktracking(size, adj)
            if gr >= 5:
                return adj, edges
    return None


def _random_k_regular(n, k, rng):
    """Generate a simple random k-regular graph via configuration model."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    for _ in range(50):
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if ok and all(len(adj[i]) == k for i in range(n)):
            return adj
    return None


def _moore_lower_bound(k, g):
    """Compute the Moore bound for k-regular graphs of girth g."""
    if g % 2 == 0:
        d = g // 2
        return 2 * (k ** (d - 1) + sum(k ** i for i in range(d - 1)))
    else:
        d = (g - 1) // 2
        return k + (k - 1) * sum(k ** i for i in range(d))


def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible.
    Uses voltage graph (cyclic cover) construction with randomized restarts."""
    start_time = time.time()
    time_budget = 110  # leave 10s margin

    best = None
    best_n = 10**9

    # Moore lower bound
    lb = _moore_lower_bound(k, g)

    # Strategy: try different base sizes and covering moduli
    # For 3-regular girth 16, base of size 6-12 with m ~ 100-200 should work
    # n = base_size * m, so we want small base * reasonable m

    base_sizes = [6, 8, 10, 12, 14, 16]
    m_values = []

    # Choose m values that give n in a reasonable range
    for bs in base_sizes:
        for m in range(40, 300, 4):
            n = bs * m
            if n >= lb and n < best_n:
                m_values.append((bs, m))

    # Sort by resulting n (ascending)
    m_values.sort(key=lambda x: x[0] * x[1])

    # Also try some specific good combinations
    extra_combos = []
    for bs in base_sizes:
        for m in [80, 100, 120, 140, 160, 180, 200, 220, 250, 280]:
            n = bs * m
            if n >= lb:
                extra_combos.append((bs, m))
    extra_combos.sort(key=lambda x: x[0] * x[1])
    m_values = extra_combos + m_values

    # Deduplicate while preserving order
    seen = set()
    unique_combos = []
    for combo in m_values:
        if combo not in seen:
            seen.add(combo)
            unique_combos.append(combo)
    m_values = unique_combos

    rng = random.Random(42)

    for bs, m in m_values:
        if time.time() - start_time > time_budget:
            break
        n_target = bs * m
        if n_target >= best_n:
            continue

        # Find a base graph of size bs with girth >= 5
        base = None
        for attempt in range(30):
            adj = _random_k_regular(bs, k, rng)
            if adj is None:
                continue
            edges = []
            for i in range(bs):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            gr = _girth_nonbacktracking(bs, adj)
            if gr >= 5:
                base = (adj, edges)
                break

        if base is None:
            continue

        base_adj, base_edges = base

        # Try voltage lifts
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
        per_try_time = min(10, remaining_time / 3)

        result = _try_voltage_lift(base_adj, base_edges, g, m, rng,
                                   time_limit=per_try_time)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best = (n, edges)

    if best is None:
        # Fallback: try direct random regular graphs
        best = _fallback_random_regular(k, g, time_budget - (time.time() - start_time))

    if best is None:
        # Ultimate fallback: use LUW construction
        best = _luw_fallback(k, g)

    return best


def _fallback_random_regular(k, g, time_budget=30):
    """Last-resort: random k-regular graph generation."""
    start_time = time.time()
    best = None
    best_n = 10**9
    rng = random.Random(12345)

    lb = _moore_lower_bound(k, g)

    for n in range(lb, min(lb + 500, 1200), 6):
        if time.time() - start_time > time_budget:
            break
        if n * k % 2 != 0:
            continue
        if n >= best_n:
            continue
        for attempt in range(50):
            adj = _random_k_regular(n, k, rng)
            if adj is None:
                continue
            gr = _girth_nonbacktracking(n, adj)
            if gr >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best is None or n < best[0]:
                    best = (n, edges)
                    best_n = n
                break

    return best


def _luw_fallback(k, g):
    """Minimal algebraic fallback (compact LUW) to guarantee a valid answer."""
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
