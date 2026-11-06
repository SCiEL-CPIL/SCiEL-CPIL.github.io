# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 150.0
SAFETY = 8.0
N = 20


def _time_left():
    return DEADLINE - time.time() - SAFETY


def _is_valid(adj):
    """Check no three vertices share three common neighbours (no K3,3)."""
    for a in range(N):
        ra = adj[a]
        for b in range(a + 1, N):
            if b not in ra:
                continue
            inter = ra & adj[b]
            if len(inter) >= 3:
                return False
    return True


def _try_add(adj, u, v):
    """Try to add edge u-v; returns True and updates adj if it stays K3,3-free."""
    ru = adj[u]
    rv = adj[v]
    if v in ru:
        return True
    inter = ru & rv
    if len(inter) >= 3:
        return False
    # Adding edge is safe iff no other edge (x,y) with x in ru, y in rv
    # (both non-adjacent to the opposite endpoint) has |N(x)&N(y)| >= 2.
    for x in ru:
        if x == v:
            continue
        rx = adj[x]
        for y in rv:
            if y == u:
                continue
            if u in rx or x in adj[y]:
                continue
            if len(rx & adj[y]) >= 2:
                return False
    ru.add(v)
    rv.add(u)
    return True


def _init_adj():
    """Build a dense valid starting graph."""
    adj = [set() for _ in range(N)]
    for u in range(N):
        for v in range(u + 1, N):
            _try_add(adj, u, v)
    return adj


def _random_adj():
    """Random valid starting graph."""
    adj = [set() for _ in range(N)]
    order = list(range(N * (N - 1) // 2))
    random.shuffle(order)
    for i in order:
        u = i // (N - 1)
        v = i - u * (N - 1)
        if v <= u:
            v += 1
        _try_add(adj, u, v)
    return adj


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _local_search(adj):
    """Greedy add, then random restart add/delete passes."""
    n = _edge_count(adj)
    while _time_left() > 0.2:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if v not in adj[u] and _try_add(adj, u, v):
                    improved = True
        if not improved:
            break
    n = _edge_count(adj)
    while _time_left() > 0.2:
        # random deletion pass
        deleted = False
        for _ in range(6):
            u = random.randrange(N)
            v = random.randrange(N)
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
                deleted = True
        if not deleted:
            break
        # greedy re-add
        for u in range(N):
            for v in range(u + 1, N):
                if v not in adj[u] and _try_add(adj, u, v):
                    pass
        if _edge_count(adj) > n:
            n = _edge_count(adj)
    return n


def construct():
    best = None
    best_n = -1
    # deterministic strong start
    adj = _init_adj()
    n = _local_search(adj)
    if n > best_n:
        best_n = n
        best = adj
    # randomized restarts
    while _time_left() > 1.0:
        adj = _random_adj()
        n = _local_search(adj)
        if n > best_n:
            best_n = n
            best = adj
    if best is None:
        best = _init_adj()
    edges = []
    for u in range(N):
        for v in best[u]:
            if u < v:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
