"""Rewritten solver: largest clique subdivision via max-flow feasibility + search."""
import time, random, heapq
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
EDGES = P['edges']
ADJ = [set() for _ in range(N)]
for a, b in EDGES:
    ADJ[a].add(b)
    ADJ[b].add(a)


def _build_flow(branches):
    """Return (nodes, adj, src, snk, need) for a unit-capacity vertex-split flow network,
    or None if a pair of branches is disconnected."""
    k = len(branches)
    pairs = list(combinations(branches, 2))
    need = len(pairs)
    nonbranch = [v for v in range(N) if v not in branches]
    # node id map
    idmap = {}
    nxt = 0
    for v in branches:
        idmap[(v, 'in')] = nxt; nxt += 1
        idmap[(v, 'out')] = nxt; nxt += 1
    for v in nonbranch:
        idmap[(v, 'in')] = nxt; nxt += 1
        idmap[(v, 'out')] = nxt; nxt += 1
    src = nxt; nxt += 1
    snk = nxt; nxt += 1
    nodes = nxt
    adj = [[] for _ in range(nodes)]

    def add(u, v, cap):
        adj[u].append([v, cap, len(adj[v])])
        adj[v].append([u, 0, len(adj[u]) - 1])

    # branch in->out with capacity = degree in pair set (>=2)
    for v in branches:
        add(idmap[(v, 'in')], idmap[(v, 'out')], 2 * k)
    # nonbranch in->out capacity 1 (internal-disjointness)
    for v in nonbranch:
        add(idmap[(v, 'in')], idmap[(v, 'out')], 1)
    # graph edges
    for a in range(N):
        ia, oa = idmap[(a, 'in')], idmap[(a, 'out')]
        for b in ADJ[a]:
            if b > a:
                add(ia, idmap[(b, 'out')], 1)
                add(idmap[(b, 'in')], oa, 1)
    # source / sink to each branch
    for idx, (u, v) in enumerate(pairs):
        add(src, idmap[(u, 'in')], 1)
        add(idmap[(v, 'out')], snk, 1)
    return nodes, adj, src, snk, need, pairs, idmap


def _maxflow(nodes, adj, src, snk):
    """Dinic. Returns max flow value and the edge flows (for reconstruction)."""
    level = [0] * nodes
    it = [0] * nodes
    parent_edge = [[-1, -1] for _ in range(nodes)]  # node -> (prev_node, edge_index_in_prev)

    def bfs():
        for i in range(nodes):
            level[i] = -1
        q = [src]
        level[src] = 0
        for i in range(nodes):
            parent_edge[i][0] = -1
            parent_edge[i][1] = -1
        qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for e in adj[u]:
                v = e[0]
                if e[1] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    parent_edge[v][0] = u
                    parent_edge[v][1] = len(adj[u]) - 1
                    q.append(v)
        return level[snk] >= 0

    def dfs(u, f):
        if u == snk:
            return f
        while it[u] < len(adj[u]):
            e = adj[u][it[u]]
            v = e[0]
            if e[1] > 0 and level[v] == level[u] + 1:
                ret = dfs(v, f if f < e[1] else e[1])
                if ret > 0:
                    e[1] -= ret
                    adj[v][e[2]][1] += ret
                    return ret
            it[u] += 1
        return 0

    flow = 0
    while bfs():
        for i in range(nodes):
            it[i] = 0
        while True:
            f = dfs(src, 10**9)
            if f == 0:
                break
            flow += f
    return flow


def _reconstruct(nodes, adj, src, snk, pairs, idmap):
    """Walk each unit of flow from src to snk; return list of vertex-paths."""
    paths = []
    # For each pair, find one s-t path in the residual (edges with flow>0).
    # We locate flow by scanning forward edges that have positive flow.
    # Build a flow graph: edge u->v with flow = reverse residual capacity.
    flow_adj = [[] for _ in range(nodes)]
    for u in range(nodes):
        for e in adj[u]:
            v = e[0]
            rev = adj[v][e[2]]
            f = rev[1]  # flow on u->v
            if f > 0:
                flow_adj[u].append(v)
    # For each pair, find an s-t path that starts with the src->uin edge of u and ends vout->snk.
    # Simpler: find all s-t paths in flow graph (each unit), then map to vertex path.
    # Use DFS enumerating paths.
    result = []
    used = [False] * len(adj)  # not needed; we just find one path per unit
    # We'll repeatedly find a path in flow_adj and remove its edges.
    import sys
    sys.setrecursionlimit(10000)
    for _ in range(len(pairs)):
        # find any s-t path in flow_adj
        stack = [(src, [src])]
        found = None
        while stack:
            u, path = stack.pop()
            if u == snk:
                found = path
                break
            for v in flow_adj[u]:
                stack.append((v, path + [v]))
        if found is None:
            return None
        # remove one unit along found path
        for a, b in zip(found, found[1:]):
            for e in adj[a]:
                if e[0] == b:
                    rev = adj[b][e[2]]
                    if rev[1] > 0:
                        rev[1] -= 1
                        e[1] += 1
                        break
        # convert node path to vertex path: nodes are (v,'in'/'out') or src/snk
        verts = []
        for nd in found:
            if nd == src or nd == snk:
                continue
            # find vertex
            for v in range(N):
                if idmap.get((v, 'in')) == nd or idmap.get((v, 'out')) == nd:
                    if not verts or verts[-1] != v:
                        verts.append(v)
                    break
        result.append(verts)
    return result


def _try_set(branches):
    """Return (paths, ok). paths maps pair->vertex list."""
    built = _build_flow(branches)
    if built is None:
        return None, False
    nodes, adj, src, snk, need, pairs, idmap = built
    flow = _maxflow(nodes, adj, src, snk)
    if flow < need:
        return None, False
    paths = _reconstruct(nodes, adj, src, snk, pairs, idmap)
    if paths is None:
        return None, False
    # map path to its pair by endpoints
    out = {}
    for p in paths:
        out[(p[0], p[-1])] = p
    for (u, v) in pairs:
        if (u, v) not in out and (v, u) not in out:
            return None, False
    return out, True


def _finalize(branches, pathmap):
    branches = [int(b) for b in branches]
    paths = []
    for u, v in combinations(branches, 2):
        p = pathmap.get((u, v)) or pathmap.get((v, u))
        paths.append([int(x) for x in p])
    return {'branches': branches, 'paths': paths}


def _validate(res):
    branches = res['branches']
    if len(branches) < 2:
        return False
    if len(set(branches)) != len(branches):
        return False
    bset = set(branches)
    used = set()
    for p in res['paths']:
        if p[0] not in bset or p[-1] not in bset:
            return False
        for i in range(len(p) - 1):
            if p[i] in bset or p[i + 1] in bset:
                return False
            if p[i + 1] not in ADJ[p[i]]:
                return False
        for x in p[1:-1]:
            if x in bset or x in used:
                return False
            used.add(x)
    return True


def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0
    rng = random.Random(12345)

    # ---- Greedy construction (fast, deterministic) ----
    # Start from a high-degree seed, grow by adding a vertex that keeps feasibility.
    def greedy(start_order):
        best = None
        best_size = 0
        cur = [start_order[0]]
        # try to grow
        cand = list(start_order[1:])
        while True:
            added = False
            for v in list(cand):
                trial = cur + [v]
                if len(trial) < 3:
                    continue
                pm, ok = _try_set(trial)
                if ok:
                    cur = trial
                    cand.remove(v)
                    added = True
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best = (list(cur), pm)
                    break
            if not added:
                break
        return best

    # order by degree desc
    order = sorted(range(N), key=lambda v: -len(ADJ[v]))
    gb = None
    for s in order[:6]:
        if time.time() > DEADLINE:
            break
        r = greedy(order)
        if r and (gb is None or len(r[0]) > len(gb[0])):
            gb = r

    # ---- Branch and bound on candidate set ----
    # We search subsets of a candidate pool (top vertices by degree + random).
    pool = sorted(set(order[:14] + [rng.randrange(N) for _ in range(10)]), key=lambda v: -len(ADJ[v]))

    best = gb
    best_size = len(best[0]) if best else 0

    def bb(node_set, remaining, depth):
        nonlocal best, best_size
        if time.time() > DEADLINE:
            return
        # bound: node_set + remaining can't beat best
        if len(node_set) + len(remaining) <= best_size:
            return
        if node_set:
            pm, ok = _try_set(node_set)
            if ok and len(node_set) > best_size:
                best_size = len(node_set)
                best = (list(node_set), pm)
        if not remaining:
            return
        for i, v in enumerate(remaining):
            if time.time() > DEADLINE:
                return
            bb(node_set + [v], remaining[i + 1:], depth + 1)

    # bounded B&B: only explore up to size best_size+2, with a node budget
    node_budget = 4000
    nodes_seen = [0]
    def bb2(node_set, remaining):
        nonlocal best, best_size
        if time.time() > DEADLINE or nodes_seen[0] > node_budget:
            return
        nodes_seen[0] += 1
        if len(node_set) + len(remaining) <= best_size:
            return
        if node_set:
            pm, ok = _try_set(node_set)
            if ok and len(node_set) > best_size:
                best_size = len(node_set)
                best = (list(node_set), pm)
        for i, v in enumerate(remaining):
            bb2(node_set + [v], remaining[i + 1:])

    # run a few random pool B&B
    for attempt in range(12):
        if time.time() > DEADLINE:
            break
        p = sorted(set(order[:10] + [rng.randrange(N) for _ in range(8)]), key=lambda v: -len(ADJ[v]))
        p = p[:12]
        nodes_seen[0] = 0
        bb2([], p)

    # ---- Random local search ----
    def random_ls():
        nonlocal best, best_size
        for _ in range(4000):
            if time.time() > DEADLINE:
                return
            size = min(N, best_size + rng.randint(0, 3))
            s = rng.sample(range(N), size)
            pm, ok = _try_set(s)
            if ok and size > best_size:
                best_size = size
                best = (list(s), pm)
            # try to grow
            if ok:
                cur = list(s)
                cand = [v for v in range(N) if v not in cur]
                rng.shuffle(cand)
                for v in cand:
                    if time.time() > DEADLINE:
                        return
                    trial = cur + [v]
                    pm2, ok2 = _try_set(trial)
                    if ok2:
                        cur = trial
                        if len(cur) > best_size:
                            best_size = len(cur)
                            best = (list(cur), pm2)
                        break

    random_ls()

    if best is None:
        # fallback: an edge
        u, v = EDGES[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    res = _finalize(best[0], best[1])
    if not _validate(res):
        # fallback to a safe small answer
        u, v = EDGES[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    return res
