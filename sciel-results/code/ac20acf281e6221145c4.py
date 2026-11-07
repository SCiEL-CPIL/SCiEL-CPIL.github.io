import random
import time

N = 15

def _check_add(adj, deg, u, v):
    # adj: list of sets
    # Case 1: uv middle: s~u, t~v, s!=t, s!=v, u!=t, deg(s)==deg(t)
    for s in adj[u]:
        if s == v:
            continue
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[s] == deg[t]:
                return False
    # Case 2: uv first: u-v-b-t, b~v, t~b, distinct u,v,b,t, deg(u)==deg(t)
    du = deg[u]
    for b in adj[v]:
        if b == u:
            continue
        for t in adj[b]:
            if t == u or t == v or t == b:
                continue
            if deg[t] == du:
                return False
    # Case 3: uv last: s-a-u-v, a~u, s~a, distinct, deg(s)==deg(v)
    dv = deg[v]
    for a in adj[u]:
        if a == v:
            continue
        for s in adj[a]:
            if s == u or s == v or s == a:
                continue
            if deg[s] == dv:
                return False
    return True

def _add_edge(adj, deg, u, v):
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1

def _remove_edge(adj, deg, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1

def _edge_count(adj):
    return sum(len(s) for s in adj) // 2

def _full_check(adj, deg):
    for u in range(N):
        for v in adj[u]:
            if u >= v:
                continue
            # recompute? we trust incremental; but verify all 3-edge paths
            pass
    # brute verify
    for u in range(N):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
            for b in adj[v]:
                if b == u:
                    continue
                for t in adj[b]:
                    if t == u or t == v or t == b:
                        continue
                    if deg[t] == deg[u]:
                        return False
            for a in adj[u]:
                if a == v:
                    continue
                for s in adj[a]:
                    if s == u or s == v or s == a:
                        continue
                    if deg[s] == deg[v]:
                        return False
    return True

def _greedy(seed, deadline):
    rng = random.Random(seed)
    adj = [set() for _ in range(N)]
    deg = [0]*N
    pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
    rng.shuffle(pairs)
    for (u,v) in pairs:
        if time.time() > deadline:
            break
        if v not in adj[u] and _check_add(adj, deg, u, v):
            _add_edge(adj, deg, u, v)
    # local search: try swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
        rng.shuffle(pairs)
        for (a,b) in pairs:
            if time.time() > deadline:
                break
            if b in adj[a]:
                continue
            if not _check_add(adj, deg, a, b):
                continue
            # try removing one existing edge to make room? we can only add if valid.
            _add_edge(adj, deg, a, b)
            improved = True
            break
    return adj, deg

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    best = None
    best_cnt = -1
    seed = 0
    while time.time() < deadline:
        adj, deg = _greedy(seed, deadline)
        if _full_check(adj, deg):
            cnt = _edge_count(adj)
            if cnt > best_cnt:
                best_cnt = cnt
                best = adj
        seed += 1
    if best is None:
        # fallback star
        adj = [set() for _ in range(N)]
        for v in range(1, N):
            _add_edge(adj, [0]*N, 0, v)
        best = adj
    edges = []
    for u in range(N):
        for v in best[u]:
            if u < v:
                edges.append([int(u), int(v)])
    return edges
