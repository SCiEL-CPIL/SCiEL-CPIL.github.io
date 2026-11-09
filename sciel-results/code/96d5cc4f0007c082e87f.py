import time
from random import Random

def _gen_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    import sys
    sys.setrecursionlimit(100000)
    N = 20
    all_edges = set()
    for a in range(N):
        for b in range(a + 1, N):
            all_edges.add((a, b))
    trees = _gen_trees(N, 611)
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v); adj[v].append(u)
            deg[u] += 1; deg[v] += 1
        maxdeg = max(deg) if deg else 0
        tree_info.append(dict(k=k, edges=edges, adj=adj, deg=deg, maxdeg=maxdeg, nedges=len(edges)))

    start = time.time()
    DEADLINE = start + 145.0

    def find_embedding(info, used, node_limit, tlimit):
        k = info['k']; edges = info['edges']; adj = info['adj']; deg = info['deg']
        if len(edges) > len(all_edges) - len(used):
            return None
        if k <= 1:
            return [0]
        avail = all_edges - used
        an = [set() for _ in range(N)]
        for (a, b) in avail:
            an[a].add(b); an[b].add(a)
        host_deg = [len(s) for s in an]
        maxdeg = info['maxdeg']
        # Hall-like necessary condition
        tree_deg_sorted = sorted(deg, reverse=True)
        host_deg_sorted = sorted(host_deg, reverse=True)
        for t in range(1, maxdeg + 1):
            ct = 0
            for d in tree_deg_sorted:
                if d >= t:
                    ct += 1
                else:
                    break
            ch = 0
            for d in host_deg_sorted:
                if d >= t:
                    ch += 1
                else:
                    break
            if ct > ch:
                return None
        t2h = [-1] * k
        h2t = [-1] * N
        nodes = [0]
        t0 = time.time()
        root = max(range(k), key=lambda v: deg[v])

        def dfs(frontier, assigned_count):
            if time.time() - t0 > tlimit:
                return False
            if assigned_count == k:
                return True
            # Forward check: ensure each frontier vertex has at least one candidate
            bestv = None; bestc = None
            for v in frontier:
                p = -1
                for nb in adj[v]:
                    if t2h[nb] != -1:
                        p = nb; break
                host = t2h[p]
                cands = [h for h in an[host] if h2t[h] == -1]
                if not cands:
                    return False
                if bestc is None or len(cands) < len(bestc):
                    bestv = v; bestc = cands
            v = bestv
            p = -1
            for nb in adj[v]:
                if t2h[nb] != -1:
                    p = nb; break
            host = t2h[p]
            for h in sorted(bestc, key=lambda x: -len(an[x])):
                t2h[v] = h; h2t[h] = v
                newfrontier = [x for x in frontier if x != v]
                for nb in adj[v]:
                    if t2h[nb] == -1:
                        newfrontier.append(nb)
                if dfs(newfrontier, assigned_count + 1):
                    return True
                t2h[v] = -1; h2t[h] = -1
            return False

        # root placement with symmetry breaking (by degree class)
        seen_deg = set()
        for h in range(N):
            if len(an[h]) in seen_deg:
                continue
            seen_deg.add(len(an[h]))
            t2h[root] = h; h2t[h] = root
            frontier = [nb for nb in adj[root]]
            if dfs(frontier, 1):
                return [int(t2h[i]) for i in range(k)]
            t2h[root] = -1; h2t[h] = -1
        return None

    def value_of(result):
        return sum(tree_info[i]['nedges'] for i in range(N) if result[i] is not None)

    def try_order(order, used, result, node_limit, tlimit):
        for ti in order:
            if time.time() > DEADLINE - 3.0:
                return
            if result[ti] is not None:
                continue
            info = tree_info[ti]
            emb = find_embedding(info, used, node_limit, tlimit)
            if emb is not None:
                for u, v in info['edges']:
                    a, b = emb[u], emb[v]
                    used.add((a, b) if a < b else (b, a))
                result[ti] = [int(x) for x in emb]

    indices = list(range(N))
    base_orders = []
    base_orders.append(sorted(indices, key=lambda i: (-tree_info[i]['nedges'], -tree_info[i]['maxdeg'])))
    base_orders.append(sorted(indices, key=lambda i: (-tree_info[i]['maxdeg'], -tree_info[i]['nedges'])))
    base_orders.append(sorted(indices, key=lambda i: (-tree_info[i]['nedges'])))
    base_orders.append(sorted(indices, key=lambda i: (-tree_info[i]['maxdeg'])))
    rr = Random(777)
    for _ in range(4):
        o = indices[:]; rr.shuffle(o); base_orders.append(o)

    best_result = [None] * N
    best_value = 0

    def record(result):
        nonlocal best_value, best_result
        val = value_of(result)
        if val > best_value:
            best_value = val
            best_result = [x[:] if x is not None else None for x in result]

    # deterministic strong attempts
    for oi in range(min(3, len(base_orders))):
        if time.time() > DEADLINE - 8.0:
            break
        order = base_orders[oi][:]
        used = set(); result = [None] * N
        tlimit = 4.0
        node_limit = 2000000
        try_order(order, used, result, node_limit, tlimit)
        record(result)
        if best_value >= 190:
            break

    # adaptive restarts
    r = Random(4242)
    oi = 3
    while time.time() < DEADLINE - 4.0:
        if best_value >= 190:
            break
        base = base_orders[oi % len(base_orders)]
        oi += 1
        order = base[:]
        for _ in range(3):
            i = r.randrange(N); j = r.randrange(N)
            order[i], order[j] = order[j], order[i]
        remaining = DEADLINE - time.time() - 4.0
        tlimit = max(0.3, min(3.5, remaining / 20.0))
        node_limit = 1200000
        used = set(); result = [None] * N
        try_order(order, used, result, node_limit, tlimit)
        record(result)

    # polish: fill gaps in best_result with strong search
    if best_value < 190 and time.time() < DEADLINE - 2.0:
        used = set()
        for i in range(N):
            if best_result[i] is not None:
                for u, v in tree_info[i]['edges']:
                    a, b = best_result[i][u], best_result[i][v]
                    used.add((a, b) if a < b else (b, a))
        for i in range(N):
            if time.time() > DEADLINE - 1.0:
                break
            if best_result[i] is None:
                emb = find_embedding(tree_info[i], used, 3000000, 8.0)
                if emb is not None:
                    for u, v in tree_info[i]['edges']:
                        a, b = emb[u], emb[v]
                        used.add((a, b) if a < b else (b, a))
                    best_result[i] = [int(x) for x in emb]
                    best_value = value_of(best_result)
                    if best_value >= 190:
                        break

    if best_result[0] is None:
        best_result[0] = [0]
    return best_result
