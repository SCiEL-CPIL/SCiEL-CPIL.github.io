"""Rewritten: strong edge colouring via MIS / clique-cover B&B + local search."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _instance(n, d, seed):
    rng = random.Random(seed)
    degree, edges = [0] * n, set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_conflict(edges):
    """Return list of neighbour bitmasks for the strong-conflict graph.
    Two edges conflict if they share a vertex or are joined by an edge (distance-2)."""
    n = len(edges)
    # map edge index
    eidx = {e: i for i, e in enumerate(edges)}
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    neigh = [0] * n
    for i, e in enumerate(edges):
        u, v = e
        near = {u, v}
        for x in (u, v):
            for y in adj[x]:
                near.add(y)
        # all edges touching any vertex in `near` conflict with e (including e itself)
        for a, b in edges:
            j = eidx[(a, b)]
            if j == i:
                continue
            if (a in near) or (b in near):
                neigh[i] |= (1 << j)
                neigh[j] |= (1 << i)
    return neigh


def _greedy_mis(neigh, mask):
    """Greedy maximum independent set on the conflict graph restricted to `mask`.
    Returns (independent_set_bitmask, count). Uses a weight-based ordering."""
    # collect candidate vertices
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        verts.append(i)
        m ^= lb
    # order by (degree within mask, index) ascending -> pick low-degree first
    degs = []
    for i in verts:
        d = (neigh[i] & mask).bit_count()
        degs.append((d, i))
    degs.sort()
    chosen = 0
    for _, i in degs:
        if (mask >> i) & 1:
            # i still available
            chosen |= (1 << i)
            mask &= ~(neigh[i] | (1 << i))
    return chosen, chosen.bit_count()


def _greedy_clique(neigh, mask):
    """Greedy clique lower bound on the conflict subgraph induced by `mask`."""
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        verts.append(i)
        m ^= lb
    # order by degree descending
    verts.sort(key=lambda i: (neigh[i] & mask).bit_count(), reverse=True)
    clique = 0
    for i in verts:
        if (mask >> i) & 1:
            # must be adjacent to all current clique members
            ok = True
            c = clique
            while c:
                lb = c & (-c)
                j = lb.bit_length() - 1
                if not ((neigh[i] >> j) & 1):
                    ok = False
                    break
                c ^= lb
            if ok:
                clique |= (1 << i)
                mask &= ~(neigh[i] | (1 << i))
    return clique.bit_count()


def _local_search(neigh, classes, max_iter=200):
    """Try to reduce number of classes via 2-opt swaps between two classes.
    classes: list of frozensets of edge indices (a partition)."""
    k = len(classes)
    improved = True
    it = 0
    while improved and it < max_iter and _remaining() > 1.0:
        improved = False
        it += 1
        for a in range(k):
            for b in range(a + 1, k):
                # try to move an edge from a to b if it fits, or swap
                # simple: move one edge from a to b
                for x in list(classes[a]):
                    if (neigh[x] & classes[b]) == 0:
                        classes[a] = frozenset(classes[a] - {x})
                        classes[b] = frozenset(classes[b] | {x})
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        # drop empty classes
        classes = [c for c in classes if c]
    return classes


def _solve_instance(edges, time_budget):
    n = len(edges)
    if n == 0:
        return []
    neigh = _build_conflict(edges)
    full = (1 << n) - 1

    # ---- Greedy upper bound: repeatedly extract a greedy MIS ----
    remaining = full
    classes = []
    while remaining:
        if _remaining() < 1.0:
            break
        s, _ = _greedy_mis(neigh, remaining)
        if s == 0:
            # fallback: single vertex
            lb = remaining & (-remaining)
            s = lb
        classes.append(frozenset(i for i in range(n) if (s >> i) & 1))
        remaining &= ~s
    # local search to reduce classes
    classes = _local_search(neigh, classes)
    best = len(classes)
    best_classes = classes

    # ---- Exact B&B if budget allows ----
    if n <= 40 and _remaining() > 3.0:
        # branch and bound on independent set cover (equivalently colour the conflict graph)
        import sys
        sys.setrecursionlimit(10000)
        state = {'best': best, 'best_classes': best_classes, 'nodes': 0}

        def clique_lb(mask):
            return _greedy_clique(neigh, mask)

        def dfs(mask, used, current_classes):
            if mask == 0:
                if used < state['best']:
                    state['best'] = used
                    state['best_classes'] = [frozenset(c) for c in current_classes]
                return True
            if _remaining() < 1.0:
                return True
            lb = clique_lb(mask)
            if used + lb >= state['best']:
                return True
            # pick a vertex to branch on: highest degree in mask
            verts = []
            m = mask
            while m:
                lbm = m & (-m)
                i = lbm.bit_length() - 1
                verts.append(i)
                m ^= lbm
            verts.sort(key=lambda i: (neigh[i] & mask).bit_count(), reverse=True)
            v = verts[0]
            # Option A: put v in a new colour class with a maximal independent set
            s, _ = _greedy_mis(neigh, mask)
            if not (s >> v) & 1:
                s |= (1 << v)
            current_classes.append(frozenset(i for i in range(n) if (s >> i) & 1))
            dfs(mask & ~s, used + 1, current_classes)
            current_classes.pop()
            # Option B: put v alone (or with a subset) - try v with each compatible vertex
            compat = mask & ~(neigh[v] | (1 << v))
            # try v alone
            dfs(mask & ~(1 << v), used + 1, current_classes)
            # try v paired with one compatible vertex
            for u in verts:
                if u == v:
                    continue
                if (compat >> u) & 1:
                    current_classes.append(frozenset({v, u}))
                    dfs(mask & ~(1 << v) & ~(1 << u), used + 1, current_classes)
                    current_classes.pop()
            return False

        dfs(full, 0, [])
        if state['best'] < best:
            best = state['best']
            best_classes = state['best_classes']

    # map classes back to colour indices following sorted edge order
    colour = [0] * n
    for ci, c in enumerate(best_classes):
        for i in c:
            colour[i] = ci
    return colour


def construct():
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _remaining() < 2.0:
            # not enough time: use trivial valid colouring (each edge its own colour)
            edges = _instance(*spec)
            out.append(list(range(len(edges))))
            continue
        edges = _instance(*spec)
        # allocate remaining time proportionally; give more to later (larger) instances
        n = len(edges)
        # simple heuristic: proportional to n
        budget = max(2.0, _remaining() * (n / 180.0))
        col = _solve_instance(edges, budget)
        out.append(col)
    return out
# EVOLVE-BLOCK-END
