import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def _subtrees(edges, k):
    # adjacency and degrees
    adj = [[] for _ in range(k)]
    deg = [0] * k
    for a, b in edges:
        adj[a].append(b); adj[b].append(a); deg[a] += 1; deg[b] += 1
    # root = vertex with max degree (tie: smallest index)
    root = max(range(k), key=lambda v: (deg[v], -v))
    # children via BFS from root
    parent = [-1] * k
    order = [root]
    parent[root] = root
    for v in order:
        for u in adj[v]:
            if parent[u] == -1:
                parent[u] = v
                order.append(u)
    # children lists
    children = [[] for _ in range(k)]
    for u in range(k):
        if parent[u] != u:
            children[parent[u]].append(u)
    # subtrees rooted at children of root
    subs = []
    for c in children[root]:
        # collect vertices in subtree of c
        verts = []
        stack = [c]
        while stack:
            x = stack.pop(); verts.append(x)
            for y in children[x]:
                stack.append(y)
        subs.append(verts)
    return root, children, subs

def construct():
    deadline = time.time() + 140.0
    trees = instance_trees()
    # order trees by k descending (T20 first) to place large ones while edges plentiful
    order = list(range(19, -1, -1))  # k = 20..1
    # available edges as set of frozensets {u,v}
    avail = set()
    for u in range(20):
        for v in range(u + 1, 20):
            avail.add((u, v))
    used = set()
    result = [None] * 20
    # adjacency of available edges for fast neighbor lookup
    # we maintain avail as set of (u,v) with u<v

    def avail_neighbors(x):
        # return list of y != x such that edge (min(x,y),max(x,y)) in avail
        out = []
        for y in range(20):
            if y == x:
                continue
            e = (x, y) if x < y else (y, x)
            if e in avail:
                out.append(y)
        return out

    def embed_tree(edges, k):
        # returns list of k distinct host vertices (image of tree vertex i) or None
        if k == 1:
            return [0]
        if k == 2:
            # find any available edge
            for e in list(avail):
                u, v = e
                return [u, v]
            return None
        root, children, subs = _subtrees(edges, k)
        # recursive placement: place root at some host, then children subtrees
        # We'll do DFS with backtracking over host assignments.

        # Precompute for each subtree its structure (root=that child)
        # We need to embed subtree rooted at c: map c->host, then its children...
        # Represent subtree by its vertex set and internal edges; but we have the
        # global tree edges. Easier: do a full DFS assigning tree-vertex -> host.

        # Build tree adjacency
        adj = [[] for _ in range(k)]
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
        deg = [len(adj[v]) for v in range(k)]
        # parent order from root
        parent = [-1] * k
        parent[root] = root
        order_list = [root]
        for v in order_list:
            for u in adj[v]:
                if parent[u] == -1:
                    parent[u] = v
                    order_list.append(u)
        # children
        ch = [[] for _ in range(k)]
        for u in range(k):
            if parent[u] != u:
                ch[parent[u]].append(u)

        # DFS assignment
        assign = [-1] * k  # tree vertex -> host
        # candidate hosts for root: all 20
        # We'll try to be clever: root should have enough available degree for its children count
        # children count = len(ch[root])
        need_deg = len(ch[root])

        def try_place(v, host):
            # check host not used by another tree vertex
            for tv in range(k):
                if tv != v and assign[tv] == host:
                    return False
            # check edges from v to already-assigned children (children assigned before parent? no, parent first)
            # Since we assign parent before children, when placing v, its parent is already placed.
            # Check edge (parent[v], v) available.
            if parent[v] != v:
                ph = assign[parent[v]]
                e = (host, ph) if host < ph else (ph, host)
                if e not in avail:
                    return False
            return True

        # We'll do iterative DFS with explicit stack for backtracking.
        # Order: process vertices in order_list (BFS/DFS from root).
        # For each vertex, choose a host.
        # To bound search, limit candidates.

        # Precompute for each vertex the set of tree vertices that are its children (already in ch).
        # When we place vertex v, we must ensure that for each child c of v,
        # there will be an available edge (host(v), host(c)). We can't check until c placed.
        # So backtracking handles it.

        # Candidate ordering for root: prefer hosts with high available degree.
        root_candidates = list(range(20))
        # sort by available degree descending
        def avail_deg(h):
            cnt = 0
            for y in range(20):
                if y == h: continue
                e = (h, y) if h < y else (y, h)
                if e in avail: cnt += 1
            return cnt
        root_candidates.sort(key=lambda h: -avail_deg(h))

        # We'll attempt multiple root choices; for each, do DFS.
        # Use recursion with a node counter to bound.
        best = [None]
        nodes = [0]
        MAX_NODES = 2000000

        def dfs(idx):
            if nodes[0] > MAX_NODES:
                return
            if time.time() > deadline:
                return
            if idx == k:
                # all assigned; verify all edges available (should be by construction)
                for a, b in edges:
                    ha, hb = assign[a], assign[b]
                    e = (ha, hb) if ha < hb else (hb, ha)
                    if e not in avail:
                        return
                best[0] = list(assign)
                return
            v = order_list[idx]
            # candidate hosts
            if v == root:
                cands = root_candidates
            else:
                ph = assign[parent[v]]
                cands = avail_neighbors(ph)
            # filter already-used hosts
            used_hosts = set(assign[:idx])  # only first idx assigned
            # Actually assign has -1 for unassigned; build set of assigned hosts
            used_hosts = set()
            for tv in range(k):
                if assign[tv] != -1 and tv != v:
                    used_hosts.add(assign[tv])
            cands = [h for h in cands if h not in used_hosts]
            # order candidates: prefer those with higher remaining available degree
            cands.sort(key=lambda h: -avail_deg(h))
            for h in cands:
                nodes[0] += 1
                assign[v] = h
                dfs(idx + 1)
                if best[0] is not None:
                    assign[v] = -1
                    return
                assign[v] = -1
            return

        dfs(0)
        return best[0]

    # Sequential packing
    for idx in order:
        k = idx + 1
        edges = trees[idx]
        if k == 1:
            result[idx] = None
            continue
        # embed
        mapping = embed_tree(edges, k)
        if mapping is None:
            # fallback: try to find any embedding via simpler greedy
            # For robustness, just leave None (invalid) but try harder
            mapping = embed_tree(edges, k)
        if mapping is None:
            # give up gracefully
            result[idx] = None
            continue
        # mark edges used
        if k >= 2:
            for a, b in edges:
                ha, hb = mapping[a], mapping[b]
                e = (ha, hb) if ha < hb else (hb, ha)
                if e in avail:
                    avail.discard(e)
                used.add(e)
        result[idx] = list(mapping)

    # Ensure types: Python int, list
    out = []
    for r in result:
        if r is None:
            out.append(None)
        else:
            out.append([int(x) for x in r])
    return out
