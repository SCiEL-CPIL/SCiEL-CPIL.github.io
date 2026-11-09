"""
Maximum clique subdivision (topological minor) finder.

Strategy (combined):
- Path verification uses a MAX-FLOW model (vertex-split network, one unit of flow
  per branch pair, capacity 1 on every non-branch vertex). This is a necessary and
  sufficient test for the existence of internally vertex-disjoint paths, and it
  also extracts the concrete paths -- far more reliable than pure greedy/backtrack.
- Branch-set search: max clique (Bron-Kerbosch) as a strong seed, then greedy grow
  (add a vertex while the flow stays feasible), then randomized local search
  (swap in/out) to escape local optima.
- An explicit deadline guard returns the best feasible result found.
"""
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_flow_disjoint_paths(adj, branches, deadline):
    """
    Decide whether internally vertex-disjoint paths exist for every unordered pair
    of branches, and if so extract them.

    Vertex-split network:
      node 2v  = v_in,  node 2v+1 = v_out,  edge (v_in -> v_out) capacity 1 for
      non-branch v (capacity INF for branch v so it can be reused as endpoint).
      For each undirected graph edge (u,v): edges (u_out -> v_in) and
      (v_out -> u_in) with capacity INF.
      A super-source S has capacity-1 edges to each branch's v_in; a super-sink T
      has capacity-1 edges from each branch's v_out.
    Sending k(k-1)/2 units of flow and routing the flow through a branch's
    v_in->v_out edge to the correct pair yields the required disjoint paths.

    Returns (ok, paths_dict) where paths_dict maps (u, v) (u < v) -> list of vertices.
    """
    n = P['n']
    k = len(branches)
    if k < 2:
        return False, {}

    INF = 10 ** 9
    S = 2 * n
    T = 2 * n + 1
    N = 2 * n + 2
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    m = len(pairs)

    # adjacency: list of [to, cap, rev_index]
    g = [[] for _ in range(N)]

    def add_edge(u, v, cap):
        g[u].append([v, cap, len(g[v])])
        g[v].append([u, 0, len(g[u]) - 1])

    for v in range(n):
        cap = INF if v in branch_set else 1
        add_edge(2 * v, 2 * v + 1, cap)
    for u, v in P['edges']:
        add_edge(2 * u + 1, 2 * v, INF)
        add_edge(2 * v + 1, 2 * u, INF)
    for b in branches:
        add_edge(S, 2 * b, 1)
        add_edge(2 * b + 1, T, 1)

    # Edmonds-Karp (BFS augmenting paths). n is tiny, so this is fast.
    flow = 0
    while flow < m:
        if time.time() > deadline:
            return False, {}
        parent = [-1] * N
        pedge = [-1] * N
        q = deque([S])
        parent[S] = S
        while q:
            u = q.popleft()
            if u == T:
                break
            for i, e in enumerate(g[u]):
                v, cap, _ = e
                if cap > 0 and parent[v] == -1:
                    parent[v] = u
                    pedge[v] = i
                    q.append(v)
        if parent[T] == -1:
            return False, {}
        # augment by 1
        v = T
        while v != S:
            u = parent[v]
            e = g[u][pedge[v]]
            e[1] -= 1
            g[v][e[2]][1] += 1
            v = u
        flow += 1

    # Extract paths: follow flow from S through each branch's in-node.
    paths_dict = {}
    for b in branches:
        b_in = 2 * b
        b_out = 2 * b + 1
        # find the used edge b_in -> b_out
        used_edge = None
        for e in g[b_in]:
            if e[0] == b_out and e[1] == INF - 1:  # originally INF, now reduced by 1
                used_edge = e
                break
        if used_edge is None:
            return False, {}
        # forward from b_out
        forward = [b]
        cur = b_out
        while True:
            nxt = None
            for e in g[cur]:
                if e[0] != T and e[1] == INF - 1 and e[2] != -1:
                    # check it is a forward flow edge (residual cap reduced)
                    rev = g[e[0]][e[2]]
                    if rev[1] == 1:  # reverse has flow -> this edge carries flow
                        nxt = e[0]
                        break
            if nxt is None:
                return False, {}
            if nxt == T:
                break
            forward.append(nxt // 2)
            cur = nxt // 2 * 2 + 1  # move to out-node of nxt
        # backward from b_in
        backward = [b]
        cur = b_in
        while True:
            nxt = None
            for e in g[cur]:
                if e[0] != S and e[2] != -1:
                    rev = g[e[0]][e[2]]
                    if rev[1] == 1 and e[1] == INF - 1:
                        nxt = e[0]
                        break
            if nxt is None:
                return False, {}
            if nxt == S:
                break
            backward.append(nxt // 2)
            cur = nxt // 2 * 2  # move to in-node of nxt
        backward = backward[::-1]  # now starts at the other branch, ends at b
        full = backward + forward[1:]
        a, z = full[0], full[-1]
        paths_dict[(a, z)] = full

    if len(paths_dict) != m:
        return False, {}
    return True, paths_dict


def _verify(adj, branches, deadline):
    """Return (ok, paths_dict) with paths_dict keyed by (min, max) vertex ids."""
    if len(branches) < 2:
        return False, {}
    ok, pd = _max_flow_disjoint_paths(adj, branches, deadline)
    if not ok:
        return False, {}
    # normalize keys
    norm = {}
    for (a, z), p in pd.items():
        key = (min(a, z), max(a, z))
        if key in norm:
            return False, {}  # duplicate pair -> invalid
        norm[key] = p
    if len(norm) != len(list(combinations(sorted(branches), 2))):
        return False, {}
    # sanity: endpoints match, interiors disjoint & avoid branches
    used = set()
    bset = set(branches)
    for (a, z), p in norm.items():
        if p[0] != a or p[-1] != z:
            return False, {}
        for x in p[1:-1]:
            if x in bset or x in used:
                return False, {}
            used.add(x)
        for x, y in zip(p, p[1:]):
            if y not in adj[x]:
                return False, {}
    return True, norm


def _max_clique(adj, deadline):
    n = P['n']
    deg = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = {v: i for i, v in enumerate(order)}
    best = []

    def bk(R, Pset, X):
        nonlocal best
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if len(R) + len(Pset) <= len(best):
            return
        PX = Pset | X
        if PX:
            pivot = max(PX, key=lambda v: len(adj[v] & Pset))
            cand = [v for v in Pset if v not in adj[pivot]]
        else:
            cand = list(Pset)
        cand.sort(key=lambda v: -deg[v])
        for v in cand:
            if time.time() > deadline:
                return
            bk(R | {v}, Pset & adj[v], X & adj[v])
            Pset.discard(v)
            X.add(v)

    # degeneracy-style: process in reverse order with P restricted to later vertices
    bk(set(), set(range(n)), set())
    if not best:
        best = [0, 1]
    return best


def _greedy_grow(adj, branches, deadline):
    n = P['n']
    bset = set(branches)
    cur = list(branches)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cands = [v for v in range(n) if v not in bset]
        cands.sort(key=lambda v: -len(adj[v]))
        for cand in cands:
            if time.time() > deadline:
                break
            trial = cur + [cand]
            ok, _ = _verify(adj, trial, deadline)
            if ok:
                cur.append(cand)
                bset.add(cand)
                improved = True
                break
    return cur


def _local_search(adj, branches, deadline):
    n = P['n']
    best = list(branches)
    best_size = len(best)
    best_paths = None
    ok, pd = _verify(adj, best, deadline)
    if ok:
        best_paths = pd
    while time.time() < deadline - 2:
        bset = set(best)
        cands = [v for v in range(n) if v not in bset]
        if not cands:
            break
        # try adding a random candidate
        if random.random() < 0.6:
            cand = random.choice(cands)
            trial = best + [cand]
        else:
            # swap: remove a random branch, add a random candidate
            if len(best) <= 2:
                cand = random.choice(cands)
                trial = best + [cand]
            else:
                rem = random.choice(best)
                cand = random.choice(cands)
                trial = [x for x in best if x != rem] + [cand]
        ok, pd = _verify(adj, trial, deadline)
        if ok and len(trial) >= best_size:
            if len(trial) > best_size:
                best = trial
                best_size = len(trial)
                best_paths = pd
    return best, best_paths


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin vs 90s wall-clock kill

    n = P['n']
    adj = _build_adj()

    # Step 1: seed with max clique
    clique = _max_clique(adj, deadline)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    branches = sorted(clique)
    ok, pd = _verify(adj, branches, deadline)
    if not ok:
        # shrink until feasible
        for size in range(len(branches) - 1, 1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in combinations(branches, size):
                ok, pd = _verify(adj, list(subset), deadline)
                if ok:
                    branches = sorted(subset)
                    found = True
                    break
            if found:
                break
        if not ok:
            u, v = P['edges'][0]
            branches = sorted([u, v])
            ok, pd = _verify(adj, branches, deadline)
            if not ok:
                return {'branches': [int(u), int(v)],
                        'paths': [[int(u), int(v)]]}

    best_branches = branches
    best_paths = pd
    best_size = len(branches)

    # Step 2: greedy grow
    if time.time() < deadline - 10:
        grown = _greedy_grow(adj, branches, deadline)
        ok, pd = _verify(adj, grown, deadline)
        if ok and len(grown) > best_size:
            best_branches = sorted(grown)
            best_paths = pd
            best_size = len(grown)
            branches = best_branches

    # Step 3: randomized local search
    if time.time() < deadline - 4:
        ls_branches, ls_paths = _local_search(adj, branches, deadline)
        if ls_paths is not None and len(ls_branches) > best_size:
            best_branches = sorted(ls_branches)
            best_paths = ls_paths
            best_size = len(ls_branches)

    # Final verification of the best result
    ok, pd = _verify(adj, best_branches, deadline)
    if ok:
        best_paths = pd

    # Build output
    paths = []
    for a, z in combinations(sorted(best_branches), 2):
        p = best_paths.get((a, z))
        if p is None:
            p = best_paths.get((z, a))
        if p is None:
            p = [a, z]
        paths.append([int(x) for x in p])

    best_branches = [int(b) for b in sorted(best_branches)]
    return {'branches': best_branches, 'paths': paths}
