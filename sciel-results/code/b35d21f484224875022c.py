# EVOLVE-BLOCK-START
import time

DEADLINE = 118.0
N = 40

def _can_add(adj, u, v):
    # no common neighbor => no C6 through (u,v)
    a = adj[u]
    b = adj[v]
    if len(a) < len(b):
        a, b = b, a
    for x in a:
        if x in b:
            return False
    return True

def _polish(adj, edges, deadline):
    changed = True
    while changed:
        changed = False
        for u in range(N):
            if time.time() > deadline:
                return
            au = adj[u]
            for v in range(u + 1, N):
                if time.time() > deadline:
                    return
                if v not in au and _can_add(adj, u, v):
                    au.add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    changed = True

def _swap_improve(adj, edges, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(N):
            au = adj[u]
            for v in list(au):
                if v <= u:
                    continue
                if time.time() > deadline:
                    return
                # remove (u,v)
                au.remove(v)
                adj[v].remove(u)
                edges.remove([u, v])
                # try to add two edges among the freed-up non-neighbours
                added = 0
                # candidates: non-neighbours of u (now including v) and of v (now including u)
                for x in range(N):
                    if x == u or x == v:
                        continue
                    if x in au:
                        continue
                    if x not in adj[v] and _can_add(adj, u, x):
                        au.add(x)
                        adj[x].add(u)
                        edges.append([u, x])
                        added += 1
                        if time.time() > deadline:
                            break
                    if added >= 2:
                        break
                for x in range(N):
                    if x == u or x == v:
                        continue
                    if x in adj[v]:
                        continue
                    if x not in au and _can_add(adj, v, x):
                        adj[v].add(x)
                        adj[x].add(v)
                        edges.append([v, x])
                        added += 1
                        if time.time() > deadline:
                            break
                    if added >= 2:
                        break
                if added > 1:
                    improved = True
                else:
                    # revert
                    au.add(v)
                    adj[v].add(u)
                    edges.append([u, v])

def construct():
    t0 = time.time()
    deadline = t0 + (DEADLINE - 2.0)

    # ---- Phase 1: greedy random construction, keep best ----
    import random
    best_edges = None
    best_count = 0

    for seed in range(0, 40000):
        if time.time() > deadline:
            break
        random.seed(seed)
        adj = [set() for _ in range(N)]
        edges = []
        order = list(range(N))
        random.shuffle(order)
        # add vertices in order, connecting to as many earlier vertices as possible
        placed = []
        for v in order:
            cands = [u for u in placed if _can_add(adj, u, v)]
            # shuffle candidates to vary structure
            random.shuffle(cands)
            for u in cands:
                if time.time() > deadline:
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
            placed.append(v)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = [s.copy() for s in adj]

    if best_edges is None:
        # fallback: 8 disjoint K5
        e = []
        for k in range(8):
            e += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return e

    adj = best_adj

    # ---- Phase 2: systematic polish (add all possible edges) ----
    _polish(adj, edges, deadline)

    # ---- Phase 3: systematic swap search for further improvement ----
    _swap_improve(adj, edges, deadline)

    # ---- Phase 4: final polish after swaps ----
    _polish(adj, edges, deadline)

    # safety: return a valid simple edge list
    seen = set()
    out = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b])
    return out
# EVOLVE-BLOCK-END
