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
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    N = 20
    trees = _gen_trees(N, 611)

    # ---- Build the 19 perfect matchings of K20 (1-factorization) ----
    # K20 is K_{2m} with m=10. Use the standard round-robin construction.
    # Fix vertex 0; the other 19 vertices are paired in 9 rounds + 1, total 10 rounds? 
    # Actually K_{2m} has 2m-1 = 19 perfect matchings.
    # Standard: vertices 0..19. Fix 0. For round r in 0..18, matching pairs
    #   (0, r) is not right. Use the circle method on 19 "moving" vertices.
    # Circle method for K_{2m}: place vertices 1..19 on a circle, fix 0.
    # Round r: pair (0, a_r) and for i=1..8: ((a_r + i) mod 19, (a_r - i) mod 19)
    # where a_r = r+1? Let's use the well-known construction.
    matchings = []
    # vertices: 0 fixed, 1..19 in a ring
    ring = list(range(1, N))  # 1..19
    for r in range(N - 1):  # 19 rounds
        M = []
        # 0 pairs with ring[r]
        M.append((0, ring[r]))
        for i in range(1, (N - 1) // 2 + 1):
            a = ring[(r + i) % (N - 1)]
            b = ring[(r - i) % (N - 1)]
            M.append((a, b))
        matchings.append(M)
    # Verify: each matching is a perfect matching and union covers K20 exactly once.
    # (We trust the construction; it's the standard 1-factorization.)

    # Precompute for each matching: partner map and edge set
    match_partner = []   # list of [partner or -1 for each host vertex]
    match_edge_set = []  # list of frozenset of (a,b) a<b
    for M in matchings:
        part = [-1] * N
        for a, b in M:
            part[a] = b; part[b] = a
        match_partner.append(part)
        es = frozenset((min(a, b), max(a, b)) for a, b in M)
        match_edge_set.append(es)

    # ---- Tree info ----
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v); adj[v].append(u)
            deg[u] += 1; deg[v] += 1
        maxdeg = max(deg) if deg else 0
        # BFS order from a leaf (root = highest degree for better search)
        if k <= 1:
            order = list(range(k))
        else:
            root = max(range(k), key=lambda v: deg[v])
            visited = [False] * k
            order = [root]; visited[root] = True
            frontier = [root]
            while frontier:
                nf = []
                for nd in frontier:
                    for nb in adj[nd]:
                        if not visited[nb]:
                            visited[nb] = True
                            order.append(nb); nf.append(nb)
                frontier = nf
            for v in range(k):
                if not visited[v]:
                    order.append(v)
        tree_info.append(dict(k=k, edges=edges, adj=adj, deg=deg, maxdeg=maxdeg,
                              nedges=len(edges), order=order))

    start = time.time()
    DEADLINE = start + 145.0

    # ---- Embed a single tree given a fixed set of allowed matchings ----
    # The tree's edges must be drawn from the allowed matchings.
    # Each tree edge (u,v) must be assigned a matching m in allowed such that
    #   partner_m(host[u]) == host[v].
    # We do a DFS over tree vertices in BFS order; when placing a vertex whose
    # parent is already placed, the matching for that edge is chosen from allowed,
    # and the host is forced = partner_m(host[parent]).
    # But a vertex may have multiple children; each child edge needs a matching
    # consistent with the vertex's host. This is a constraint satisfaction.

    def find_embedding(info, used, node_limit, tlimit):
        k = info['k']; edges = info['edges']; adj = info['adj']; deg = info['deg']
        nedges = info['nedges']
        if k <= 1:
            return [0]
        if nedges == 0:
            return [0]
        # available matchings: those not fully used? We track used edges, not matchings.
        # A matching m is "available for an edge (a,b)" iff (a,b) not in used.
        # We'll track which matchings have been partially consumed.
        # Simpler: allowed matchings = all 19, but an edge (a,b) can only use matching m
        # if (a,b) is in matching m AND (a,b) not in used.
        # Since each edge of K20 belongs to exactly one matching, the matching is
        # determined by the edge! So "assign tree edge to a matching" is automatic:
        # tree edge (u,v) -> host edge (h1,h2) -> matching m(h1,h2).
        # The constraint is just that all host edges are distinct and unused.
        # So the matching structure reduces the *choice*: for a placed parent at host h,
        # and a child, the child's host must be partner_m(h) for SOME matching m,
        # i.e., any of the 19 partners. That's the same as "any unused neighbor".
        # So the matching view doesn't reduce branching for a single tree.
        # BUT it does give a global structure: we can pre-assign each tree a *subset*
        # of matchings to use, and then the tree's edges are constrained to that subset.
        # This is the key: assign disjoint matching-subsets to trees.

        # For now, within a tree, the host edge for (u,v) is any unused edge.
        # We use the standard frontier DFS with forward checking.
        avail = [(a, b) for a in range(N) for b in range(a + 1, N) if (a, b) not in used]
        an = [set() for _ in range(N)]
        for (a, b) in avail:
            an[a].add(b); an[b].add(a)
        host_deg = [len(s) for s in an]
        maxdeg = info['maxdeg']
        tree_deg_sorted = sorted(deg, reverse=True)
        host_deg_sorted = sorted(host_deg, reverse=True)
        for t in range(1, maxdeg + 1):
            ct = sum(1 for d in tree_deg_sorted if d >= t)
            ch = sum(1 for d in host_deg_sorted if d >= t)
            if ct > ch:
                return None
        if nedges > len(avail):
            return None

        t2h = [-1] * k
        h2t = [-1] * N
        nodes = [0]
        t0 = time.time()
        root = max(range(k), key=lambda v: deg[v])

        def dfs(frontier, assigned_count):
            if nodes[0] > node_limit or time.time() - t0 > tlimit:
                return False
            if assigned_count == k:
                return True
            bestv = None; bestc = None; bestscore = None
            for v in frontier:
                p = -1
                for nb in adj[v]:
                    if t2h[nb] != -1:
                        p = nb; break
                host = t2h[p]
                cands = []
                for h in an[host]:
                    if h2t[h] == -1:
                        cands.append(h)
                if not cands:
                    return False
                score = (len(cands), -deg[v])
                if bestscore is None or score < bestscore:
                    bestv = v; bestc = cands; bestscore = score
            v = bestv
            p = -1
            for nb in adj[v]:
                if t2h[nb] != -1:
                    p = nb; break
            host = t2h[p]
            for h in sorted(bestc, key=lambda x: len(an[x])):
                nodes[0] += 1
                t2h[v] = h; h2t[h] = v
                newfrontier = [x for x in frontier if x != v]
                for nb in adj[v]:
                    if t2h[nb] == -1:
                        newfrontier.append(nb)
                ok = True
                for x in newfrontier:
                    if x == v:
                        continue
                    pp = -1
                    for nb2 in adj[x]:
                        if t2h[nb2] != -1:
                            pp = nb2; break
                    hh = t2h[pp]
                    has_cand = False
                    for h2 in an[hh]:
                        if h2t[h2] == -1:
                            has_cand = True
                            break
                    if not has_cand:
                        ok = False
                        break
                if ok and dfs(newfrontier, assigned_count + 1):
                    return True
                t2h[v] = -1; h2t[h] = -1
            return False

        seen = {}
        for h in range(N):
            key = frozenset(an[h])
            if key in seen:
                continue
            seen[key] = True
            nodes[0] += 1
            t2h[root] = h; h2t[h] = root
            frontier = [nb for nb in adj[root]]
            if dfs(frontier, 1):
                return [int(t2h[i]) for i in range(k)]
            t2h[root] = -1; h2t[h] = -1
        return None

    # ---- Global strategy: assign disjoint matching-subsets to trees, then embed ----
    # Each tree T_k needs e_k = k-1 edges, drawn from its assigned matchings.
    # A matching provides at most 1 edge per tree (since a tree is connected and
    # uses each matching at most... actually a tree could use multiple edges from
    # the same matching? No: a matching is a set of disjoint edges; a tree can use
    # several edges from the same matching as long as they don't share a host vertex
    # in the embedding. But since the tree's vertices map to distinct hosts, and a
    # matching has each host in exactly one edge, a tree can use at most ... it can
    # use multiple edges from one matching if those matching-edges are disjoint in the
    # host graph, which they always are. So a tree could in principle use many edges
    # from one matching. But for a *simple* packing we want each tree to use edges
    # from distinct matchings to keep it clean. Actually the cleanest: each tree uses
    # exactly one edge per matching it's assigned, and we assign it exactly e_k matchings.
    # Then the tree's e_k edges come from e_k distinct matchings, one each.
    # This is a strong constraint that makes embedding easy: for each tree edge, the
    # matching is fixed, so the host edge is fixed up to which matching-edge within
    # the matching maps to it.

    # Let's assign matchings to trees: tree k needs e_k = k-1 matchings.
    # Total needed = sum(k-1) = 190 = 19*10. But we only have 19 matchings, each can
    # be used by multiple trees (a matching has 10 edges, so up to 10 trees can each
    # use 1 edge from it). Total edge-slots across matchings = 19*10 = 190. 
    # So we need to assign to each tree k a set of e_k matchings, such that each
    # matching is assigned to at most 10 trees (since it has 10 edges).
    # This is a bipartite matching / flow problem: trees (demand e_k) to matchings
    # (capacity 10 each). Total demand 190 = total capacity 190, so it's tight.

    # Greedy assignment: process trees in decreasing e_k, assign them to matchings
    # with the most remaining capacity.
    def assign_matchings():
        cap = [10] * 19  # each matching can serve 10 trees
        assign = [None] * N  # assign[k-1] = list of matching indices
        # order trees by e_k descending
        order = sorted(range(N), key=lambda i: -tree_info[i]['nedges'])
        for i in order:
            e = tree_info[i]['nedges']
            if e == 0:
                assign[i] = []
                continue
            # pick e matchings with highest remaining cap
            cands = sorted(range(19), key=lambda m: -cap[m])
            chosen = cands[:e]
            for m in chosen:
                cap[m] -= 1
            assign[i] = chosen
        return assign

    def find_embedding_constrained(info, used, node_limit, tlimit, allowed_matchings):
        """Embed tree using only edges from allowed_matchings (one edge per matching)."""
        k = info['k']; edges = info['edges']; adj = info['adj']; deg = info['deg']
        nedges = info['nedges']
        if k <= 1:
            return [0]
        if nedges == 0:
            return [0]
        # Build allowed host edges: for each matching in allowed_matchings, the 10 edges,
        # but only those not in used.
        allowed_edges = []
        for m in allowed_matchings:
            for (a, b) in match_edge_set[m]:
                if (a, b) not in used:
                    allowed_edges.append((a, b))
        if len(allowed_edges) < nedges:
            return None
        an = [set() for _ in range(N)]
        for (a, b) in allowed_edges:
            an[a].add(b); an[b].add(a)
        host_deg = [len(s) for s in an]
        maxdeg = info['maxdeg']
        tree_deg_sorted = sorted(deg, reverse=True)
        host_deg_sorted = sorted(host_deg, reverse=True)
        for t in range(1, maxdeg + 1):
            ct = sum(1 for d in tree_deg_sorted if d >= t)
            ch = sum(1 for d in host_deg_sorted if d >= t)
            if ct > ch:
                return None
        if nedges > len(allowed_edges):
            return None
        t2h = [-1] * k
        h2t = [-1] * N
        nodes = [0]
        t0 = time.time()
        root = max(range(k), key=lambda v: deg[v])

        def dfs(frontier, assigned_count):
            if nodes[0] > node_limit or time.time() - t0 > tlimit:
                return False
            if assigned_count == k:
                return True
            bestv = None; bestc = None; bestscore = None
            for v in frontier:
                p = -1
                for nb in adj[v]:
                    if t2h[nb] != -1:
                        p = nb; break
                host = t2h[p]
                cands = []
                for h in an[host]:
                    if h2t[h] == -1:
                        cands.append(h)
                if not cands:
                    return False
                score = (len(cands), -deg[v])
                if bestscore is None or score < bestscore:
                    bestv = v; bestc = cands; bestscore = score
            v = bestv
            p = -1
            for nb in adj[v]:
                if t2h[nb] != -1:
                    p = nb; break
            host = t2h[p]
            for h in sorted(bestc, key=lambda x: len(an[x])):
                nodes[0] += 1
                t2h[v] = h; h2t[h] = v
                newfrontier = [x for x in frontier if x != v]
                for nb in adj[v]:
                    if t2h[nb] == -1:
                        newfrontier.append(nb)
                ok = True
                for x in newfrontier:
                    if x == v:
                        continue
                    pp = -1
                    for nb2 in adj[x]:
                        if t2h[nb2] != -1:
                            pp = nb2; break
                    hh = t2h[pp]
                    has_cand = False
                    for h2 in an[hh]:
                        if h2t[h2] == -1:
                            has_cand = True
                            break
                    if not has_cand:
                        ok = False
                        break
                if ok and dfs(newfrontier, assigned_count + 1):
                    return True
                t2h[v] = -1; h2t[h] = -1
            return False

        seen = {}
        for h in range(N):
            key = frozenset(an[h])
            if key in seen:
                continue
            seen[key] = True
            nodes[0] += 1
            t2h[root] = h; h2t[h] = root
            frontier = [nb for nb in adj[root]]
            if dfs(frontier, 1):
                return [int(t2h[i]) for i in range(k)]
            t2h[root] = -1; h2t[h] = -1
        return None

    def value_of(result):
        return sum(tree_info[i]['nedges'] for i in range(N) if result[i] is not None)

    best_result = [None] * N
    best_value = 0

    def record(result):
        nonlocal best_value, best_result
        val = value_of(result)
        if val > best_value:
            best_value = val
            best_result = [x[:] if x is not None else None for x in result]

    # ---- Strategy 1: constrained packing via matching assignment ----
    # Try multiple random matching assignments; for each, embed trees in an order.
    r = Random(12345)
    attempts = 0
    while time.time() < DEADLINE - 6.0 and best_value < 190 and attempts < 40:
        attempts += 1
        # randomize the assignment by shuffling matching capacities / order
        cap = [10] * 19
        # randomize which trees get which matchings: shuffle the tree order
        order = list(range(N))
        r.shuffle(order)
        # sort by e_k descending for stability but with random tie-breaking
        order = sorted(order, key=lambda i: (-tree_info[i]['nedges'], r.random()))
        assign = [None] * N
        for i in order:
            e = tree_info[i]['nedges']
            if e == 0:
                assign[i] = []
                continue
            cands = sorted(range(19), key=lambda m: (-cap[m], r.random()))
            chosen = cands[:e]
            for m in chosen:
                cap[m] -= 1
            assign[i] = chosen
        # embed in order: largest first
        embed_order = sorted(range(N), key=lambda i: -tree_info[i]['nedges'])
        used = set(); result = [None] * N
        for ti in embed_order:
            if time.time() > DEADLINE - 4.0:
                break
            info = tree_info[ti]
            emb = find_embedding_constrained(info, used, 500000, 1.5, assign[ti])
            if emb is not None:
                for u, v in info['edges']:
                    a, b = emb[u], emb[v]
                    used.add((a, b) if a < b else (b, a))
                result[ti] = [int(x) for x in emb]
        record(result)

    # ---- Strategy 2: unconstrained backtracking (current program's strength) ----
    # Fill in any gaps using the general DFS.
    if best_value < 190 and time.time() < DEADLINE - 3.0:
        used = set()
        for i in range(N):
            if best_result[i] is not None:
                for u, v in tree_info[i]['edges']:
                    a, b = best_result[i][u], best_result[i][v]
                    used.add((a, b) if a < b else (b, a))
        # embed remaining in descending size
        remaining = [i for i in range(N) if best_result[i] is None]
        remaining.sort(key=lambda i: -tree_info[i]['nedges'])
        for i in remaining:
            if time.time() > DEADLINE - 1.0:
                break
            emb = find_embedding(tree_info[i], used, 2000000, 4.0)
            if emb is not None:
                for u, v in tree_info[i]['edges']:
                    a, b = emb[u], emb[v]
                    used.add((a, b) if a < b else (b, a))
                best_result[i] = [int(x) for x in emb]
                best_value = value_of(best_result)
                if best_value >= 190:
                    break

    # ---- Polish: try to fill any remaining gaps with constrained search ----
    if best_value < 190 and time.time() < DEADLINE - 1.0:
        used = set()
        for i in range(N):
            if best_result[i] is not None:
                for u, v in tree_info[i]['edges']:
                    a, b = best_result[i][u], best_result[i][v]
                    used.add((a, b) if a < b else (b, a))
        for i in range(N):
            if time.time() > DEADLINE - 0.5:
                break
            if best_result[i] is None:
                emb = find_embedding(tree_info[i], used, 3000000, 2.0)
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
    for i in range(N):
        if best_result[i] is not None:
            best_result[i] = [int(x) for x in best_result[i]]
    return best_result
