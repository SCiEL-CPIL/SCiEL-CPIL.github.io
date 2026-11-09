import time
import random

N = 20
DEADLINE = 120.0
MARGIN = 20.0


def _is_k23_free(adj):
    for u in range(N):
        for v in range(u + 1, N):
            if len(adj[u] & adj[v]) >= 3:
                return False
    return True


def _common_count(adj, u, v):
    return len(adj[u] & adj[v])


def _add_edge(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)


def _remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def _edge_list(adj):
    out = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                out.append([u, v])
    return out


def _greedy(adj):
    order = list(range(N * (N - 1) // 2))
    random.shuffle(order)
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    for idx in order:
        u, v = pairs[idx]
        if v in adj[u]:
            continue
        if _common_count(adj, u, v) < 3:
            _add_edge(adj, u, v)


def _try_swap(adj, remove, add):
    a, b = remove
    c, d = add
    if b in adj[a]:
        _remove_edge(adj, a, b)
    if d in adj[c]:
        _remove_edge(adj, c, d)
    _add_edge(adj, c, d)
    _add_edge(adj, a, b)
    if _is_k23_free(adj):
        return True
    # revert
    _remove_edge(adj, a, b)
    _remove_edge(adj, c, d)
    if b in adj[a]:
        _remove_edge(adj, a, b)
    if d in adj[c]:
        _remove_edge(adj, c, d)
    _add_edge(adj, a, b)
    _add_edge(adj, c, d)
    return False


def _local_search(adj):
    n_edges = sum(len(s) for s in adj) // 2
    while time.time() < DEADLINE - MARGIN:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if v in adj[u]:
                    continue
                if _common_count(adj, u, v) >= 3:
                    continue
                # try to add (u,v); if it creates a K2,3 pair, maybe fix by removing one edge
                _add_edge(adj, u, v)
                if _is_k23_free(adj):
                    n_edges += 1
                    improved = True
                else:
                    # find a pair (x,y) that is now at 3 common neighbors due to (u,v)
                    # remove an edge to restore
                    fixed = False
                    # the new K2,3 involves u,v as the "3 common" side? Actually adding (u,v)
                    # increases common-neighbor count for pairs that include u or v.
                    # Try removing one of the edges incident to u or v that is redundant.
                    cand = []
                    for w in list(adj[u]):
                        if w == v:
                            continue
                        cand.append((u, w))
                    for w in list(adj[v]):
                        if w == u:
                            continue
                        cand.append((v, w))
                    random.shuffle(cand)
                    for (x, y) in cand:
                        if _try_swap(adj, (x, y), (u, v)):
                            n_edges += 1
                            improved = True
                            fixed = True
                            break
                    if not fixed:
                        _remove_edge(adj, u, v)
        # 2-opt: swap an existing edge for a missing one to enable more
        existing = []
        missing = []
        for i in range(N):
            for j in range(i + 1, N):
                if j in adj[i]:
                    existing.append((i, j))
                else:
                    missing.append((i, j))
        random.shuffle(existing)
        random.shuffle(missing)
        for (a, b) in existing:
            for (c, d) in missing:
                if {a, b} == {c, d}:
                    continue
                if _try_swap(adj, (a, b), (c, d)):
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return n_edges


def construct():
    start = time.time()
    best_edges = 0
    best_adj = [set() for _ in range(N)]

    # baseline: greedy
    adj = [set() for _ in range(N)]
    _greedy(adj)
    n_edges = sum(len(s) for s in adj) // 2
    if n_edges > best_edges:
        best_edges = n_edges
        best_adj = [set(s) for s in adj]

    # local search from greedy
    adj = [set(s) for s in best_adj]
    n_edges = _local_search(adj)
    if n_edges > best_edges:
        best_edges = n_edges
        best_adj = [set(s) for s in adj]

    # multiple random restarts
    while time.time() < DEADLINE - MARGIN:
        adj = [set() for _ in range(N)]
        _greedy(adj)
        n_edges = _local_search(adj)
        if n_edges > best_edges:
            best_edges = n_edges
            best_adj = [set(s) for s in adj]
        if time.time() > DEADLINE - MARGIN:
            break

    return _edge_list(best_adj)
