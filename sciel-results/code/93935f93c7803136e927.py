import random
import time

N = 20
C = 20
DEADLINE = 135.0  # safety margin under 150s kill


def _validate(edges):
    # edges: list of (u, v, c) with u < v
    # returns (ok, bad_count)
    adj = [[None] * N for _ in range(N)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
    bad = 0
    for a in range(N):
        for b in range(a + 1, N):
            ca = adj[a][b]
            if ca is None:
                continue
            for c in range(b + 1, N):
                cb = adj[b][c]
                if cb is None:
                    continue
                for d in range(c + 1, N):
                    cd = adj[c][d]
                    if cd is None:
                        continue
                    da = adj[a][d]
                    if da is None:
                        continue
                    if len({ca, cb, cd, da}) < 4:
                        bad += 1
    return bad == 0, bad


def _build_adj(edges):
    adj = [[None] * N for _ in range(N)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj


def _bad_c4s(adj):
    """Return list of 4-cycles (a,b,c,d) with a<b<c<d that are not rainbow."""
    res = []
    for a in range(N):
        for b in range(a + 1, N):
            ca = adj[a][b]
            if ca is None:
                continue
            for c in range(b + 1, N):
                cb = adj[b][c]
                if cb is None:
                    continue
                for d in range(c + 1, N):
                    cd = adj[c][d]
                    if cd is None:
                        continue
                    da = adj[a][d]
                    if da is None:
                        continue
                    if len({ca, cb, cd, da}) < 4:
                        res.append((a, b, c, d))
    return res


def _edge_color_map(edges):
    m = {}
    for u, v, c in edges:
        m[(min(u, v), max(u, v))] = c
    return m


def _try_add_edge(ecm, adj, u, v):
    """Try to add edge (u,v). Returns (new_ecm, new_adj) if valid, else None."""
    u, v = min(u, v), max(u, v)
    if (u, v) in ecm:
        return None
    # find a colour that (a) is not used on edges incident to u or v
    # (b) does not create a non-rainbow 4-cycle
    used = set()
    for w in range(N):
        if w == u or w == v:
            continue
        cu = ecm.get((min(u, w), max(u, w)))
        if cu is not None:
            used.add(cu)
        cv = ecm.get((min(v, w), max(v, w)))
        if cv is not None:
            used.add(cv)
    candidates = [c for c in range(C) if c not in used]
    for c in candidates:
        # check 4-cycles that include (u,v)
        # a 4-cycle containing edge (u,v): u-v-x-y-u or u-v-y-x-u
        # i.e. for each pair (x,y) with x!=u,v, y!=u,v, x!=y:
        # cycle u-v-x-y-u requires edges (v,x),(x,y),(y,u) all present
        # cycle u-v-y-x-u requires edges (v,y),(y,x),(x,u) all present
        ok = True
        for x in range(N):
            if x == u or x == v:
                continue
            cvx = ecm.get((min(v, x), max(v, x)))
            if cvx is None:
                continue
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                cxy = ecm.get((min(x, y), max(x, y)))
                if cxy is None:
                    continue
                cyu = ecm.get((min(y, u), max(y, u)))
                if cyu is None:
                    continue
                # cycle u-v-x-y-u
                if len({c, cvx, cxy, cyu}) < 4:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            new_ecm = dict(ecm)
            new_ecm[(u, v)] = c
            new_adj = _build_adj([(a, b, cc) for (a, b), cc in new_ecm.items()])
            return new_ecm, new_adj
    return None


def _local_search(ecm, adj, time_limit):
    """Try to re-color edges to allow more additions. Returns (ecm, adj, improved)."""
    t_end = time.time() + time_limit
    ecm = dict(ecm)
    adj = _build_adj([(a, b, c) for (a, b), c in ecm.items()])
    # current bad count
    bad = len(_bad_c4s(adj))
    best = bad
    # try changing colour of a random edge to reduce bad cycles
    edge_list = list(ecm.keys())
    if not edge_list:
        return ecm, adj, False
    for _ in range(200000):
        if time.time() > t_end:
            break
        if best == 0:
            break
        (u, v) = random.choice(edge_list)
        old_c = ecm[(u, v)]
        # try a new colour not used on incident edges
        used = set()
        for w in range(N):
            if w == u or w == v:
                continue
            cu = ecm.get((min(u, w), max(u, w)))
            if cu is not None:
                used.add(cu)
            cv = ecm.get((min(v, w), max(v, w)))
            if cv is not None:
                used.add(cv)
        cands = [c for c in range(C) if c not in used and c != old_c]
        if not cands:
            continue
        new_c = random.choice(cands)
        ecm[(u, v)] = new_c
        adj2 = _build_adj([(a, b, c) for (a, b), c in ecm.items()])
        bad2 = len(_bad_c4s(adj2))
        if bad2 < best:
            best = bad2
            adj = adj2
        else:
            ecm[(u, v)] = old_c
    return ecm, adj, True


def _greedy_fill(ecm, adj, deadline):
    """Greedily add edges in random order until no more can be added or deadline."""
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    for (u, v) in all_pairs:
        if time.time() > deadline:
            break
        res = _try_add_edge(ecm, adj, u, v)
        if res is not None:
            ecm, adj = res
    return ecm, adj


def construct():
    start = time.time()
    deadline = start + DEADLINE
    random.seed(12345)

    best_edges = None
    best_count = -1

    # Baseline: C20 cycle with distinct colours -> valid, 20 edges
    base = [(i, (i + 1) % N, i) for i in range(N)]
    base_ecm = {(min(u, v), max(u, v)): c for u, v, c in base}
    base_adj = _build_adj(base)
    assert _validate(base)[0], "baseline C20 must be valid"
    best_edges = list(base)
    best_count = len(base)

    # Multiple restarts: start from C20, greedy fill, then local search + more fill
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        # fresh start from C20 (with random colour assignment / vertex relabelling)
        perm = list(range(N))
        random.shuffle(perm)
        # build cycle using perm order
        cycle = []
        for i in range(N):
            a = perm[i]
            b = perm[(i + 1) % N]
            u, v = min(a, b), max(a, b)
            cycle.append((u, v, i))
        ecm = {(min(u, v), max(u, v)): c for u, v, c in cycle}
        adj = _build_adj(cycle)
        # verify baseline valid
        ok, bad = _validate(cycle)
        if not ok:
            continue

        # Phase 1: greedy fill
        ecm, adj = _greedy_fill(ecm, adj, deadline - 3)
        cur_count = len(ecm)
        if cur_count > best_count:
            best_count = cur_count
            best_edges = [(a, b, c) for (a, b), c in ecm.items()]

        # Phase 2: local search to unblock, then more greedy
        if time.time() < deadline - 8:
            ecm, adj, _ = _local_search(ecm, adj, min(3.0, max(0.5, deadline - 10)))
            ecm, adj = _greedy_fill(ecm, adj, deadline - 2)
            cur_count = len(ecm)
            if cur_count > best_count:
                best_count = cur_count
                best_edges = [(a, b, c) for (a, b), c in ecm.items()]

        # Phase 3: second local search pass for densification
        if time.time() < deadline - 6:
            ecm, adj, _ = _local_search(ecm, adj, min(2.0, max(0.3, deadline - 8)))
            ecm, adj = _greedy_fill(ecm, adj, deadline - 1)
            cur_count = len(ecm)
            if cur_count > best_count:
                best_count = cur_count
                best_edges = [(a, b, c) for (a, b), c in ecm.items()]

    # Final safety: ensure validity of best
    if best_edges is None:
        best_edges = list(base)
    ok, bad = _validate(best_edges)
    if not ok:
        # fall back to known-valid baseline
        best_edges = list(base)

    # Convert to required format [[u, v, c], ...]
    return [[u, v, c] for u, v, c in best_edges]
