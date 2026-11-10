"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)

_DEADLINE = _time.time() + 80.0
_INF = 1 << 30


def _max_clique():
    """Maximum clique via branch-and-bound with greedy coloring upper bound."""
    best = []

    def color_sort(order):
        color = {}
        for v in order:
            for c in range(1, len(order) + 1):
                if all((order[j] in color and color[order[j]] != c)
                       for j in range(len(order)) if order[j] not in _ADJ[v]):
                    color[v] = c
                    break
            else:
                color[v] = len(order)
        return color

    def expand(cand):
        nonlocal best
        if not cand:
            if len(best) < 0:
                pass
            return
        if len(best) < len(cand) + len(best) - len(cand):
            pass
        # bound: greedy coloring gives upper bound on clique size in cand
        order = list(cand)
        color = color_sort(order)
        # prune if even coloring bound can't beat best
        if len(best) >= max(color.values()) + 0 and False:
            return
        # simple upper bound: len(best) vs len(best)+... use max color
        if max(color.values()) + 0 <= len(best) and False:
            return
        # order by color ascending to prune
        order.sort(key=lambda v: color[v])
        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            # upper bound: len(best)+1? we build clique incrementally
            if len(best) + i + 1 <= len(best):
                break
            if len(best) >= len(cand):
                break
            newcand = [x for x in order[:i] if x in _ADJ[v]]
            if not newcand:
                # candidate clique = best + [v]
                if len(best) + 1 > len(best):
                    pass
                cur = best + [v]
                if len(cur) > len(best):
                    best = cur
                continue
            # coloring bound on newcand
            nc = color_sort(newcand)
            if len(best) + 1 + max(nc.values()) <= len(best):
                continue
            expand(newcand)
            if len(best) + 1 + i <= len(best):
                break

    expand(list(range(_N)))
    return best


def _build_flow(n, adj):
    """Split each vertex v into v*2 (in) and v*2+1 (out), capacity 1 on v->v'.
    Undirected edges become two directed arcs capacity INF. Returns (nodes, arcs, cap, rev)."""
    nodes = n * 2
    arcs = [[] for _ in range(nodes)]
    cap = [[] for _ in range(nodes)]
    rev = [[] for _ in range(nodes)]

    def add(u, v, c):
        arcs[u].append(len(arcs[v]))
        arcs[v].append(len(arcs[u]))
        cap[u].append(c)
        cap[v].append(0)

    for v in range(n):
        add(v * 2, v * 2 + 1, 1)
    for u in range(n):
        for v in _ADJ[u]:
            if v > u:
                add(u * 2 + 1, v * 2, _INF)
                add(v * 2 + 1, u * 2, _INF)
    return nodes, arcs, cap


def _maxflow(nodes, arcs, cap, s, t):
    """Edmonds-Karp. Returns (flow, parent_node, parent_arc)."""
    flow = 0
    while True:
        parent = [-1] * nodes
        q = deque([s])
        parent[s] = s
        while q:
            u = q.popleft()
            for i, v in enumerate(arcs[u]):
                if cap[u][i] > 0 and parent[v] == -1:
                    parent[v] = u
                    if v == t:
                        break
                    q.append(v)
            if parent[t] != -1:
                break
        if parent[t] == -1:
            break
        # find bottleneck
        addf = _INF
        v = t
        while v != s:
            u = parent[v]
            idx = arcs[u].index(v)
            addf = min(addf, cap[u][idx])
            v = u
        # augment
        v = t
        while v != s:
            u = parent[v]
            idx = arcs[u].index(v)
            cap[u][idx] -= addf
            cap[v][arcs[v][idx]] += addf
            v = u
        flow += addf
    return flow, parent


def _extract_paths(n, arcs, cap, s, t, branches):
    """From residual, extract one path per branch pair via flow decomposition."""
    # Build reverse map: for each arc index i in arcs[u] going to v, find the
    # original undirected edge (u_orig, v_orig) if this is a graph edge arc.
    # We'll walk in the residual using saturated/used flow.
    # Instead: find all unit flows by walking from each branch-out node.
    # Simpler: reconstruct by scanning arcs that carry flow (cap[v][rev] > 0 means flow on u->v).
    # We'll build a directed multigraph of used arcs (flow>0) and decompose.
    used = [[] for _ in range(n * 2)]  # used[u] = list of v with flow
    for u in range(n * 2):
        for i, v in enumerate(arcs[u]):
            f = cap[v][arcs[v][i]]  # flow on u->v equals reverse residual
            if f > 0:
                used[u].append((v, f))
    # Decompose: repeatedly find a path from any branch-out to any branch-in.
    # Map node -> original vertex
    def orig(node):
        return node // 2
    # We need paths between distinct branch pairs. Use flow conservation.
    # Build adjacency of used arcs (capacities) and walk.
    import collections as _c
    used_adj = _c.defaultdict(list)
    used_cap = {}
    for u in range(n * 2):
        for (v, f) in used[u]:
            used_adj[u].append(v)
            used_cap[(u, v)] = f
    # Walk: start from each branch's out-node, follow used arcs until reaching a branch in-node.
    # To avoid reusing, decrement used_cap.
    paths = {}
    for b in branches:
        start = b * 2 + 1
        # find all flows leaving start
        for (v, f) in used[start]:
            for _ in range(f):
                # walk
                path_nodes = [start]
                cur = start
                while True:
                    # pick next
                    nxt = None
                    for w in used_adj[cur]:
                        if used_cap.get((cur, w), 0) > 0:
                            nxt = w
                            break
                    if nxt is None:
                        break
                    used_cap[(cur, nxt)] -= 1
                    path_nodes.append(nxt)
                    cur = nxt
                    if cur // 2 in branches and cur % 2 == 0:
                        break
                # path_nodes: start(b_out) ... end(c_in)
                # convert to vertex sequence
                verts = [orig(path_nodes[0])]
                for nd in path_nodes[1:]:
                    verts.append(orig(nd))
                # remove consecutive duplicates
                clean = [verts[0]]
                for x in verts[1:]:
                    if x != clean[-1]:
                        clean.append(x)
                a, b2 = clean[0], clean[-1]
                key = tuple(sorted((a, b2)))
                if key not in paths:
                    paths[key] = clean
    return paths


def _bfs_path(avoid, a, b):
    """Shortest path from a to b avoiding 'avoid' in interiors (a,b allowed)."""
    if a == b:
        return [a]
    prev = {a: None}
    q = deque([a])
    while q:
        u = q.popleft()
        if u == b:
            break
        for w in _ADJ[u]:
            if w in prev:
                continue
            if w not in (a, b) and w in avoid:
                continue
            prev[w] = u
            q.append(w)
    if b not in prev:
        return None
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _route(branches, avoid_all):
    """Route edge-disjoint paths for all pairs via max flow. Return dict of paths or None."""
    # Build flow network
    nodes, arcs, cap = _build_flow(_N, _ADJ)
    # Super source/sink? We need all-pairs disjoint. Instead do it pair by pair
    # with increasing avoid set. Use flow per pair: s=a_out, t=b_in, cap 1 on vertices.
    paths = {}
    used_interior = set()
    # Order pairs by shortest path length to route easy ones first
    pairs = []
    for a, b in combinations(branches, 2):
        p = _bfs_path(avoid_all | used_interior, a, b)
        if p is None:
            return None
        pairs.append((len(p), a, b))
    pairs.sort()
    for _, a, b in pairs:
        p = _bfs_path(avoid_all | used_interior, a, b)
        if p is None:
            return None
        paths[(a, b)] = p
        for x in p[1:-1]:
            used_interior.add(x)
    return paths


def _route_flow(branches, avoid_all):
    """Use max-flow per pair to guarantee vertex-disjointness via unit capacities."""
    paths = {}
    used_interior = set()
    pairs = list(combinations(branches, 2))
    # order by BFS length
    order = []
    for a, b in pairs:
        p = _bfs_path(avoid_all | used_interior, a, b)
        order.append((len(p) if p else 10**6, a, b))
    order.sort()
    for _, a, b in order:
        # max flow from a to b with vertex capacities, avoiding used_interior
        nodes, arcs, cap = _build_flow(_N, _ADJ)
        # remove used interiors by cutting their in->out edge
        for x in used_interior:
            # set capacity of x*2 -> x*2+1 to 0
            i = arcs[x * 2].index(x * 2 + 1)
            cap[x * 2][i] = 0
            cap[x * 2 + 1][arcs[x * 2 + 1][i]] = 0
        s = a * 2 + 1
        t = b * 2
        flow, parent = _maxflow(nodes, arcs, cap, s, t)
        if flow < 1:
            return None
        # extract one path from parent
        path_nodes = []
        cur = t
        while cur != s:
            path_nodes.append(cur)
            u = parent[cur]
            cur = u
        path_nodes.append(s)
        path_nodes.reverse()
        verts = [orig for orig in (nd // 2 for nd in path_nodes)]
        clean = [verts[0]]
        for x in verts[1:]:
            if x != clean[-1]:
                clean.append(x)
        paths[(a, b)] = clean
        for x in clean[1:-1]:
            used_interior.add(x)
    return paths


def construct():
    global _DEADLINE
    # 1. Find a large clique
    clique = _max_clique()
    # 2. Try to route; if fails, shrink the branch set greedily
    branches = list(clique)
    paths = _route_flow(branches, set())
    # If routing failed, drop vertices until it works
    while paths is None and len(branches) > 2:
        # remove the branch with fewest options
        branches = branches[:-1]
        paths = _route_flow(branches, set())
    if paths is None or len(branches) < 2:
        # fallback to an edge
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    # Build output: one path per unordered pair
    out_paths = []
    for a, b in combinations(branches, 2):
        key = (a, b)
        out_paths.append([int(x) for x in paths[key]])
    return {'branches': [int(x) for x in branches], 'paths': out_paths}
# EVOLVE-BLOCK-END
