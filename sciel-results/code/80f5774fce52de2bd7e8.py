"""Max valid edges on 15 vertices: no 3-edge path s-u-v-t with deg(s)==deg(t)."""
import time
import random

N = 15
M = N * (N - 1) // 2
EDGE_U = [0] * M
EDGE_V = [0] * M
for _k in range(M):
    i = _k // (N - 1 - _k // (N - 1))
    j = i + 1 + _k % (N - 1 - _k // (N - 1))
    EDGE_U[_k] = i
    EDGE_V[_k] = j

def _can_add(adj, deg, u, v):
    """Is adding edge (u,v) (currently absent) valid?"""
    nu = adj[u]
    nv = adj[v]
    for s in nu:
        if s == v:
            continue
        ds = deg[s]
        for t in nv:
            if t == u or t == s:
                continue
            if deg[t] == ds:
                return False
    for s in nv:
        if s == u:
            continue
        ds = deg[s]
        for t in nu:
            if t == v or t == s:
                continue
            if deg[t] == ds:
                return False
    return True

def _can_remove(adj, deg, u, v):
    """Is removing edge (u,v) (currently present) valid?"""
    du = deg[u] - 1
    dv = deg[v] - 1
    nu = adj[u]
    nv = adj[v]
    for s in nu:
        if s == v:
            continue
        ds = deg[s]
        for t in nv:
            if t == u or t == s:
                continue
            dt = deg[t]
            if s == u:
                ds = du
            if t == v:
                dt = dv
            if ds == dt:
                return False
    for s in nv:
        if s == u:
            continue
        ds = deg[s]
        for t in nu:
            if t == v or t == s:
                continue
            dt = deg[t]
            if s == v:
                ds = dv
            if t == u:
                dt = du
            if ds == dt:
                return False
    return True

def _can_swap(adj, deg, a, b, c, d):
    """Can we remove (a,b) and add (c,d) (both currently present/absent)?"""
    da = deg[a] - 1
    db = deg[b] - 1
    dc = deg[c] + 1
    dd = deg[d] + 1
    adj[a].discard(b)
    adj[b].discard(a)
    adj[c].add(d)
    adj[d].add(c)
    deg[a] -= 1
    deg[b] -= 1
    deg[c] += 1
    deg[d] += 1
    ok = _check_validity(adj, deg)
    adj[a].add(b)
    adj[b].add(a)
    adj[c].discard(d)
    adj[d].discard(c)
    deg[a] += 1
    deg[b] += 1
    deg[c] -= 1
    deg[d] -= 1
    return ok

def _check_validity(adj, deg):
    for u in range(N):
        nu = adj[u]
        for v in nu:
            if u >= v:
                continue
            nv = adj[v]
            for s in nu:
                if s == v:
                    continue
                ds = deg[s]
                for t in nv:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
            for s in nv:
                if s == u:
                    continue
                ds = deg[s]
                for t in nu:
                    if t == v or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def _add(adj, deg, u, v):
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1

def _remove(adj, deg, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1

def _count(adj):
    return sum(len(adj[u]) for u in range(N)) // 2

def _local_search_add(adj, deg, deadline):
    """Repeatedly add any valid missing edge. Return count."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if v in adj[u]:
                    continue
                if _can_add(adj, deg, u, v):
                    _add(adj, deg, u, v)
                    improved = True
                    break
            if improved:
                break
    return _count(adj)

def _swap_search(adj, deg, deadline):
    """Try single edge-flip swaps (remove one, add one) to improve count."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        cur = _count(adj)
        present = [(u, v) for u in range(N) for v in adj[u] if u < v]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        for (a, b) in present:
            if time.time() > deadline:
                break
            for (c, d) in missing:
                if c == a or c == b or d == a or d == b:
                    continue
                if _can_swap(adj, deg, a, b, c, d):
                    _remove(adj, deg, a, b)
                    _add(adj, deg, c, d)
                    improved = True
                    break
            if improved:
                break
    return _count(adj)

def construct():
    deadline = time.time() + 82.0
    best_adj = None
    best_deg = None
    best_count = -1

    def consider(adj, deg, count):
        nonlocal best_adj, best_deg, best_count
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]

    # --- Strategy 1: complete bipartite seeds K_{p, N-p} ---
    for p in range(1, N):
        if time.time() > deadline - 8:
            break
        adj = [set() for _ in range(N)]
        deg = [0] * N
        for i in range(p):
            for j in range(p, N):
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
        count = _count(adj)
        if _check_validity(adj, deg):
            consider(adj, deg, count)
            adj, deg, count = _local_search_add(adj, deg, deadline - 8)
            consider(adj, deg, count)

    # --- Strategy 2: random greedy + polish ---
    for seed in range(40):
        if time.time() > deadline - 8:
            break
        rng = random.Random(seed * 104729 + 17)
        adj = [set() for _ in range(N)]
        deg = [0] * N
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(edges)
        for (u, v) in edges:
            if _can_add(adj, deg, u, v):
                _add(adj, deg, u, v)
        count = _count(adj)
        consider(adj, deg, count)
        adj, deg, count = _local_search_add(adj, deg, deadline - 8)
        consider(adj, deg, count)

    # --- Strategy 3: perturb best, then add + swap ---
    if best_adj is not None:
        for it in range(40):
            if time.time() > deadline - 6:
                break
            rng = random.Random(it * 7919 + 101)
            adj = [set(s) for s in best_adj]
            deg = best_deg[:]
            present = [(u, v) for u in range(N) for v in adj[u] if u < v]
            nr = min(len(present), rng.randint(2, 6))
            for (u, v) in rng.sample(present, nr):
                _remove(adj, deg, u, v)
            adj, deg, count = _local_search_add(adj, deg, deadline - 6)
            consider(adj, deg, count)
            if time.time() < deadline - 3:
                adj, deg, count = _swap_search(adj, deg, deadline - 3)
                consider(adj, deg, count)

    # --- Final exhaustive polish on the best solution ---
    if best_adj is not None and time.time() < deadline - 2:
        adj = [set(s) for s in best_adj]
        deg = best_deg[:]
        adj, deg, count = _local_search_add(adj, deg, deadline - 2)
        consider(adj, deg, count)
        if time.time() < deadline - 1:
            adj, deg, count = _swap_search(adj, deg, deadline - 1)
            consider(adj, deg, count)

    if best_adj is None:
        return []
    edges = [[u, v] for u in range(N) for v in best_adj[u] if u < v]
    return edges
