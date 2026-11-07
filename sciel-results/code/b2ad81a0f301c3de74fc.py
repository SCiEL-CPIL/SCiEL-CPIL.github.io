"""Improved program for packing trees T1..T20 into K20."""
from random import Random


def _instance_trees(n=20, seed=611):
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _embed_tree(edges, n, used_edges, rng, max_attempts=200):
    """
    Try to embed a tree (given by edges on vertices 0..k-1) into K_n
    using a randomized greedy leaf-peeling / backtracking approach.
    Returns a list of length k with the host vertex for each tree vertex,
    or None if embedding fails.
    """
    k = max((a for a, b in edges), (b for a, b in edges), default=-1) + 1
    if k == 1:
        return [rng.randrange(n)]
    if k == 2:
        a, b = edges[0]
        for _ in range(max_attempts):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u != v and (u, v) not in used_edges and (v, u) not in used_edges:
                res = [0] * k
                res[a] = u
                res[b] = v
                return res
        return None

    # Build adjacency of the tree
    adj = [set() for _ in range(k)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # Find leaves and do a topological ordering (peeling)
    from collections import deque
    deg = [len(adj[i]) for i in range(k)]
    leaves = deque([i for i in range(k) if deg[i] == 1])
    order = []
    remaining = deg[:]
    while leaves:
        v = leaves.popleft()
        order.append(v)
        for nb in list(adj[v]):
            remaining[nb] -= 1
            if remaining[nb] == 1:
                leaves.append(nb)
    # order has all k vertices; reverse so we embed from leaves inward
    order.reverse()

    for attempt in range(max_attempts):
        assignment = [None] * k
        # Pick a random root: the last vertex in order (a central vertex)
        root = order[-1]
        # Actually order[-1] is the last peeled, which should be one of the final 1-2 vertices
        # Let's just pick the vertex that appears last in the peeling order
        # For a tree, the last one or two vertices in peeling are the "core"
        # Pick root as the vertex with max degree, tie-break randomly
        max_deg = max(deg)
        candidates = [i for i in range(k) if deg[i] == max_deg]
        root = rng.choice(candidates)
        assignment[root] = rng.randrange(n)

        # BFS/DFS from root, embedding children as we go
        # We need to embed in an order where each vertex (except root) is embedded
        # after its parent. Use the peeling order: vertices peeled earlier are leaves.
        # Reverse peeling order gives: root(s) first, then their children, etc.
        # Actually, let's do a proper BFS from root.
        parent = [-1] * k
        visited = [False] * k
        queue = [root]
        visited[root] = True
        for v in queue:
            for nb in sorted(adj[v], key=lambda x: rng.random()):
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = v
                    queue.append(nb)

        # Now queue is in BFS order from root. Embed in that order.
        success = True
        for v in queue:
            if v == root:
                continue
            p = parent[v]
            host_p = assignment[p]
            # Choose a random host vertex for v that is adjacent to host_p in K_n
            # (i.e., any vertex != host_p) and such that edge (host_p, host_v) not used
            chosen = None
            for _ in range(n - 1):
                cand = rng.randrange(n)
                if cand == host_p:
                    continue
                if (min(host_p, cand), max(host_p, cand)) in used_edges:
                    continue
                chosen = cand
                break
            if chosen is None:
                success = False
                break
            assignment[v] = chosen

        if success:
            # Verify all edges are valid and not used
            ok = True
            new_edges = []
            for a, b in edges:
                ha, hb = assignment[a], assignment[b]
                if ha is None or hb is None or ha == hb:
                    ok = False
                    break
                e = (min(ha, hb), max(ha, hb))
                if e in used_edges:
                    ok = False
                    break
                new_edges.append(e)
            if ok:
                return assignment
    return None


def construct():
    trees = _instance_trees(20, 611)
    used_edges = set()
    result = []
    rng = Random(42)

    for idx in range(20):
        k = idx + 1
        edges = trees[idx]
        if k == 1:
            # Single vertex tree: assign any vertex
            result.append([0])
            continue

        embedded = _embed_tree(edges, 20, used_edges, rng, max_attempts=500)
        if embedded is None:
            # Try with more attempts and different seeds
            for seed2 in range(1, 10):
                rng2 = Random(42 + seed2)
                embedded = _embed_tree(edges, 20, used_edges, rng2, max_attempts=1000)
                if embedded is not None:
                    break
        if embedded is None:
            result.append(None)
            continue

        # Verify all vertices are distinct
        if len(set(embedded)) != k:
            result.append(None)
            continue

        # Add edges to used set
        for a, b in edges:
            ha, hb = embedded[a], embedded[b]
            used_edges.add((min(ha, hb), max(ha, hb)))

        result.append(list(embedded))

    # Ensure all entries are Python ints
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in entry])

    return final_result
