"""Combined program: repair-based local search + greedy vertex growth for R(4,6)."""

import random
import time

_START = time.time()
# Evaluator kills at 150s; leave a generous safety margin.
_DEADLINE = _START + 110.0

_RNG = random.Random(20240607)


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Exact checks (bitmask adjacency: adj[i] is an int bitmask of neighbours of i)
# ---------------------------------------------------------------------------

def _find_k4(adj, n):
    """Return a 4-tuple of vertices forming a K4, or None if K4-free."""
    for u in range(n):
        au = adj[u]
        # neighbours of u that are greater than u (avoids duplicate work)
        vv = au & (~((1 << (u + 1)) - 1))
        while vv:
            v = (vv & -vv).bit_length() - 1
            vv &= vv - 1
            # common neighbours of u and v, greater than v
            c = au & adj[v] & (~((1 << (v + 1)) - 1))
            if not c:
                continue
            cc = c
            while cc:
                x = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                # a neighbour of x inside c gives the 4th vertex
                ymask = adj[x] & c
                if ymask:
                    y = (ymask & -ymask).bit_length() - 1
                    return [u, v, x, y]
    return None


def _find_clique(adj, n, k):
    """Return a k-clique (list of k vertices) in the graph (adj,n), or None.

    Ordered backtracking with a cardinality bound for early pruning.
    """
    full = (1 << n) - 1
    found = [None]

    def rec(cand, start, chosen):
        if found[0] is not None:
            return
        if len(chosen) == k:
            found[0] = list(chosen)
            return
        need = k - len(chosen)
        # only candidates with index >= start are still usable
        usable = cand & (~((1 << start) - 1))
        if usable.bit_count() < need:
            return
        c = usable
        while c:
            v = (c & -c).bit_length() - 1
            c &= c - 1
            chosen.append(v)
            rec(cand & adj[v], v + 1, chosen)
            chosen.pop()
            if found[0] is not None:
                return

    rec(full, 0, [])
    return found[0]


def _find_indep6(adj, n):
    """Return an independent set of 6 vertices in (adj,n), or None if alpha < 6.

    An independent set in G is a clique in the complement of G.
    """
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]
    return _find_clique(comp, n, 6)


def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_indep6(adj, n) is None


# ---------------------------------------------------------------------------
# Local search / repair
# ---------------------------------------------------------------------------

def _repair(adj, n, max_iters=300000):
    """Flip random edges to eliminate K4s and independent 6-sets.

    Returns (adj, n, ok) where ok is True iff the resulting graph is valid.
    """
    for _ in range(max_iters):
        if _time_left() <= 5.0:
            break
        k4 = _find_k4(adj, n)
        if k4 is not None:
            # remove a random edge inside the K4
            a, b = _RNG.sample(k4, 2)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            # add a random edge inside the independent set
            a, b = _RNG.sample(i6, 2)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            continue
        # no violation found -> valid
        return adj, n, True
    return adj, n, _is_valid(adj, n)


def _init_random(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _RNG.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _grow(adj, n, q):
    """Try to add one vertex with random neighbourhood (prob q) and re-repair.

    Returns (adj, n, ok).
    """
    new_n = n + 1
    new_adj = adj + [0]
    for i in range(n):
        if _RNG.random() < q:
            new_adj[i] |= (1 << new_n)
            new_adj[new_n] |= (1 << i)
    new_adj, new_n, ok = _repair(new_adj, new_n)
    return new_adj, new_n, ok


# ---------------------------------------------------------------------------
# Portfolio driver
# ---------------------------------------------------------------------------

def _search_portfolio():
    best = None  # (n, adj)

    # Phase 1: repair random starts at a range of (n, density).
    # Densities tuned so that both K4s and independent 6-sets are plentiful to
    # start with (giving the repair loop work) yet close to feasible.
    plan = []
    for n in (30, 28, 26, 24, 22, 20, 18, 16):
        for p in (0.4, 0.45, 0.5, 0.55, 0.6):
            plan.append((n, p))
    _RNG.shuffle(plan)

    for n, p in plan:
        if _time_left() <= 15.0:
            break
        if best is not None and n <= best[0]:
            continue
        adj = _init_random(n, p)
        adj, n2, ok = _repair(adj, n, max_iters=50000)
        if ok and (best is None or n2 > best[0]):
            best = (n2, adj)

    # Phase 2: grow the best valid graph vertex by vertex while it succeeds.
    if best is not None and _time_left() > 10.0:
        n, adj = best
        # try a few neighbourhood densities for the new vertex
        for q in (0.5, 0.4, 0.6, 0.45, 0.55):
            if _time_left() <= 10.0:
                break
            for _ in range(8):
                if _time_left() <= 10.0:
                    break
                new_adj, new_n, ok = _grow(adj, n, q)
                if ok and new_n > n:
                    adj, n = new_adj, new_n
                # else: keep the previous best and try a different neighbourhood

    return best


# ---------------------------------------------------------------------------
# Fallback: the seed Paley graph on 17 vertices (known valid).
# ---------------------------------------------------------------------------

def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}


def _to_dict(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def construct():
    result = None
    try:
        best = _search_portfolio()
        if best is not None:
            n, adj = best
            # Exact final verification before returning.
            if _is_valid(adj, n):
                result = _to_dict(n, adj)
    except Exception:
        result = None

    if result is None:
        result = _paley17()

    return result
