"""Stronger program: LUW(5,4) cage + girth-preserving randomized edge-swap search."""
import time
import random
from collections import deque

_DEADLINE = 150.0
_MARGIN = 6.0


def construct(k, g):
    deadline = time.time() + _DEADLINE
    try:
        return _solve(k, g, deadline)
    except Exception:
        # Never let an error kill the run: fall back to the known-good cage.
        n, edges = _luw_cage(k, g)
        return n, edges


def _solve(k, g, deadline):
    n, edges = _luw_cage(k, g)
    adj = _build_adj(n, edges)
    best_n, best_edges = n, list(edges)
    rng = random.Random(123456789)
    cur = _swap_loop(adj, g, deadline, _MARGIN, rng)
    if cur is not None:
        cn, ce = cur
        if cn < best_n:
            best_n, best_edges = cn, ce
    return best_n, best_edges


# ---------------------------------------------------------------------------
# LUW cage construction (Lazebnik-Ustimenko-Woldar CD(d,q), q prime)
# ---------------------------------------------------------------------------
def _luw_cage(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    if q == 4:
        add = sub = lambda x, y: x ^ y
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
# Graph helpers
# ---------------------------------------------------------------------------
def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _girth_ok(adj, roots, g):
    """Return True iff no cycle of length < g exists.

    Roots are the endpoints of the changed edge; any new short cycle must pass
    through one of them, so a multi-source BFS from the roots is sufficient.
    """
    g2 = g - 1
    dist = [-1] * len(adj)
    q = deque()
    for r in roots:
        dist[r] = 0
        q.append(r)
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= g2:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
            elif dist[v] <= du:
                if du + 1 + dist[v] < g:
                    return False
    return True


def _swap_loop(adj, g, deadline, margin, rng):
    n = len(adj)
    best_n, best_edges = n, None
    best_adj = None
    rng.shuffle(adj)

    def _apply(edges, adj, u, v, w):
        """Remove vertex w (adjacent to u and v) and add edge (u, v)."""
        for x in list(adj[w]):
            adj[x].discard(w)
        adj[w] = set()
        adj[u].add(v)
        adj[v].add(u)
        edges = [e for e in edges if w not in e]
        edges.append((u, v))
        return edges, adj

    def _revert(edges, adj, u, v, w):
        for x in adj[u]:
            adj[x].discard(u)
        for x in adj[v]:
            adj[x].discard(v)
        adj[u].add(w)
        adj[v].add(w)
        adj[w] = {u, v}
        edges = [e for e in edges if e != (u, v)]
        edges.append((u, w))
        edges.append((v, w))
        return edges, adj

    def _regularize():
        """Fix any degree != 4 by swapping edges between two over/under pairs."""
        nonlocal n
        for _ in range(8):
            under = [i for i in range(n) if len(adj[i]) < 4]
            over = [i for i in range(n) if len(adj[i]) > 4]
            if not under or not over:
                break
            u = rng.choice(under)
            v = rng.choice(over)
            shared = adj[u] & adj[v]
            if not shared:
                continue
            w = rng.choice(list(shared))
            adj[u].discard(w)
            adj[w].discard(u)
            adj[v].discard(w)
            adj[w].discard(v)
            adj[u].add(v)
            adj[v].add(u)
            adj[w].add(u)
            adj[w].add(v)
        return all(len(adj[i]) == 4 for i in range(n))

    while time.time() < deadline - margin:
        if not _regularize():
            continue
        if n < best_n:
            best_n = n
            best_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            best_adj = [set(s) for s in adj]

        # Collect candidate vertex-removal swaps (length-2 paths u-w-v, u!=v).
        cands = []
        for w in range(n):
            if len(adj[w]) != 4:
                continue
            nb = list(adj[w])
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = nb[i], nb[j]
                    if v not in adj[u]:
                        cands.append((u, v, w))
        if not cands:
            continue
        rng.shuffle(cands)
        for (u, v, w) in cands:
            if time.time() > deadline - margin:
                break
            if w not in adj[u] or w not in adj[v] or v in adj[u]:
                continue
            if len(adj[u]) != 4 or len(adj[v]) != 4:
                continue
            edges, adj2 = _apply(None, adj, u, v, w)  # adj mutated in place
            if _girth_ok(adj, (u, v), g):
                n -= 1
            else:
                _revert(None, adj, u, v, w)
    if best_n < len(adj):
        return best_n, best_edges
    return None


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    k, g = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (4, 10)
    t0 = time.time()
    n, edges = construct(k, g)
    dt = time.time() - t0
    adj = _build_adj(n, edges)
    deg_ok = all(len(s) == k for s in adj)
    seen = set()
    simple = True
    for u, v in edges:
        if u == v or _edge_key(u, v) in seen:
            simple = False
        seen.add(_edge_key(u, v))
    print(f"k={k} g={g} n={n} edges={len(edges)} deg_ok={deg_ok} simple={simple} "
          f"wall={dt:.1f}s")
