"""Construct a 4-regular girth-10 graph via heuristic tree growth + completion."""
import random
from collections import deque

def construct(k, g):
    # We target the cage problem n(4,10). The evaluator passes k and g;
    # we build a k-regular graph of girth at least g.
    # Strategy: grow a BFS tree of radius floor((g-1)/2) from a root,
    # enforcing that no added edge creates a cycle shorter than g.
    # Then complete to k-regular by adding edges between vertices whose
    # mutual distance is at least g (so no short cycle is created).
    rng = random.Random(12345)

    # Radius of the tree: a cycle formed by two vertices at tree-depths a,b
    # connected by a new edge has length a + b + 1 (going up to LCA and down).
    # To guarantee girth >= g, we need a + b + 1 >= g for every added edge.
    # In a pure BFS tree, the LCA of two vertices at depths a and b is at
    # depth <= min(a,b), so the cycle length is a + b + 1 (if LCA is root) or
    # smaller. To be safe, we require a + b + 1 >= g for any edge we add.
    radius = (g - 1) // 2  # for g=10, radius=4

    # Build the tree: vertex 0 is root.
    # Each vertex stores: parent, depth, list of children.
    parent = {0: -1}
    depth = {0: 0}
    children = {0: []}
    order = [0]

    # Grow the tree to the desired radius, branching k ways at each level
    # (root has k children, internal nodes have k-1 children to keep degree k
    #  within the tree; the remaining degree is filled by cross-edges later).
    # Actually for a k-regular graph grown as a tree, the root has k children,
    # and each non-leaf internal node has (k-1) children (one edge used by parent).
    # Leaves at radius will get their remaining edges from completion.
    #
    # We grow level by level.
    current_level = [0]
    for lvl in range(radius):
        next_level = []
        for v in current_level:
            # number of children for v
            if v == 0:
                n_children = k
            else:
                n_children = k - 1
            for _ in range(n_children):
                new_v = len(order)
                order.append(new_v)
                parent[new_v] = v
                depth[new_v] = lvl + 1
                children[new_v] = []
                children[v].append(new_v)
                next_level.append(new_v)
        current_level = next_level

    n = len(order)
    # Build adjacency list (tree edges only for now).
    adj = {i: set() for i in range(n)}
    for v in range(1, n):
        p = parent[v]
        adj[v].add(p)
        adj[p].add(v)

    # Helper: BFS distance between two vertices (unweighted, in current graph).
    def bfs_dist(s, t, adj_local):
        if s == t:
            return 0
        visited = {s}
        dq = deque([s])
        d = 0
        while dq:
            d += 1
            for _ in range(len(dq)):
                u = dq.popleft()
                for w in adj_local[u]:
                    if w == t:
                        return d
                    if w not in visited:
                        visited.add(w)
                        dq.append(w)
        return -1  # disconnected

    # Helper: would adding edge (u,v) create a cycle shorter than g?
    # A cycle through (u,v) has length = dist(u,v) + 1 where dist is the
    # shortest path between u and v in the graph WITHOUT the new edge.
    # If that shortest path length is < g - 1, then adding the edge creates
    # a cycle of length < g. Also need to ensure u and v are not already adjacent.
    def creates_short_cycle(u, v, adj_local, girth):
        if v in adj_local[u]:
            return True  # parallel edge
        d = bfs_dist(u, v, adj_local)
        if d < 0:
            # disconnected; adding edge connects components, no cycle
            return False
        return (d + 1) < girth

    # Completion: add edges to make every vertex degree exactly k.
    # We add edges only between vertices whose current distance is >= g - 1
    # (so the new cycle is >= g). Greedily pick such pairs.
    # To keep the graph connected and avoid getting stuck, we prioritize
    # pairs that are "far apart" in the tree (different branches).
    #
    # We'll iteratively find vertices with degree < k and try to connect them
    # to other low-degree vertices far away.
    #
    # Candidate generation: for each pair (u,v) with u<v, not adjacent,
    # and creates_short_cycle is False, add the edge if both have degree < k.
    #
    # To be efficient, precompute a list of "safe" pairs.
    # For n up to a few hundred, O(n^2) BFS is fine within 120 s.

    # Compute all safe pairs (u, v) where adding edge (u,v) doesn't create a
    # cycle < g and u != v and not already adjacent.
    # We'll re-check dynamically as the graph grows, but a static safe set
    # (computed on the tree) is a good start: in the tree, dist(u,v) is the
    # tree distance. Adding cross-edges only increases distances or keeps them,
    # so a pair safe in the tree remains safe later? Not exactly—adding edges
    # can create shorter paths. So we must re-check. We'll do a dynamic greedy.

    # Greedy completion:
    # Repeat until all degrees == k or no safe move exists.
    # For efficiency, maintain a list of vertices with degree < k.
    def try_complete(adj_local, k, girth, rng_local, max_iter=200000):
        for it in range(max_iter):
            # find vertices with degree < k
            low = [v for v in range(len(adj_local)) if len(adj_local[v]) < k]
            if not low:
                return True
            # For each low vertex, find a safe partner among low vertices
            # (prefer low to balance degrees).
            rng_local.shuffle(low)
            moved = False
            for u in low:
                if len(adj_local[u]) >= k:
                    continue
                # candidates: other low vertices, then all others
                candidates = [v for v in low if v != u and len(adj_local[v]) < k]
                rng_local.shuffle(candidates)
                # also include non-low as fallback
                for v in candidates:
                    if creates_short_cycle(u, v, adj_local, girth):
                        continue
                    # add edge
                    adj_local[u].add(v)
                    adj_local[v].add(u)
                    moved = True
                    break
                if moved:
                    break
            if not moved:
                # Try harder: allow any v (not just low)
                for u in low:
                    if len(adj_local[u]) >= k:
                        continue
                    allv = [v for v in range(len(adj_local)) if v != u]
                    rng_local.shuffle(allv)
                    for v in allv:
                        if creates_short_cycle(u, v, adj_local, girth):
                            continue
                        adj_local[u].add(v)
                        adj_local[v].add(u)
                        moved = True
                        break
                    if moved:
                        break
            if not moved:
                return False
        return all(len(adj_local[v]) == k for v in range(len(adj_local)))

    # Try completion with several random seeds to increase success chance.
    success = False
    for seed in range(50):
        adj_try = {i: set(adj[i]) for i in range(n)}
        rng_try = random.Random(seed)
        if try_complete(adj_try, k, g, rng_try, max_iter=200000):
            success = True
            adj = adj_try
            break

    if not success:
        # Fallback: return the tree with a star completion that may violate girth,
        # but we must return something. As a last resort, return a known small
        # valid graph if we can build one, else raise.
        # For k=4, g=10, the tree of radius 4 has 1 + 4 + 12 + 36 + 108 = 161 vertices.
        # Completion may fail. We'll try a denser tree (radius 3) which has
        # 1 + 4 + 12 + 36 = 53 vertices, easier to complete but girth risk.
        # Actually radius must be >= (g-1)//2 = 4 to guarantee girth by tree distance.
        # If completion fails, we try a different branching or accept a larger tree.
        # As a safety net, return the incomplete graph is invalid; instead we
        # build a simple circulant-like graph and hope girth holds, else raise.
        # We'll attempt a larger radius tree (radius 5) to give more completion room.
        for extra_radius in range(1, 4):
            r2 = radius + extra_radius
            parent2 = {0: -1}
            depth2 = {0: 0}
            children2 = {0: []}
            order2 = [0]
            cur = [0]
            for lvl in range(r2):
                nxt = []
                for v in cur:
                    nc = k if v == 0 else k - 1
                    for _ in range(nc):
                        nv = len(order2)
                        order2.append(nv)
                        parent2[nv] = v
                        depth2[nv] = lvl + 1
                        children2[nv] = []
                        children2[v].append(nv)
                        nxt.append(nv)
                cur = nxt
            n2 = len(order2)
            adj2 = {i: set() for i in range(n2)}
            for v in range(1, n2):
                p = parent2[v]
                adj2[v].add(p)
                adj2[p].add(v)
            for seed in range(30):
                adj_try = {i: set(adj2[i]) for i in range(n2)}
                rng_try = random.Random(seed * 7 + 1)
                if try_complete(adj_try, k, g, rng_try, max_iter=300000):
                    adj = adj_try
                    n = n2
                    success = True
                    break
            if success:
                break

    if not success:
        # Last resort: return a trivial valid small graph only if it meets girth.
        # For k=4, g=10, no small trivial graph works. Raise to signal failure
        # (scores 0), but better than returning invalid.
        raise RuntimeError("Failed to construct a valid k-regular girth-g graph")

    # Convert adjacency to edge list.
    edges = []
    seen = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                seen.add((u, v))

    # Self-check: verify degree and girth via non-backtracking walks.
    # (The evaluator will re-check, but we avoid returning invalid graphs.)
    # Degree check
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            raise RuntimeError("Degree violation")

    # Girth check via BFS from each vertex (shortest cycle through v).
    # A cycle shorter than g exists iff for some edge (u,v), dist(u,v) in
    # graph minus that edge is < g - 1. We already enforced this during
    # construction, but double-check on a sample to save time.
    # Full check: for each vertex, BFS to find shortest cycle containing it.
    # This is O(n * (n + m)) which for n~300 is fine.
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    def shortest_cycle_through(v0, adj_l, limit):
        # BFS in the graph, but track parent to detect cycles.
        # Standard method: for each neighbor, BFS avoiding backtracking.
        best = limit + 1
        for start in adj_l[v0]:
            # BFS from start, not going back to v0 until we find a cycle
            visited = {v0: -1}  # dist from v0
            dq = deque([(start, 0)])
            visited[start] = 1
            while dq:
                u, d = dq.popleft()
                if d + 1 >= best:
                    continue
                for w in adj_l[u]:
                    if w == v0 and d + 1 < best:
                        best = d + 2  # cycle: v0 - ... - u - w(=v0)? No.
                        # Actually path v0->start->...->u->w where w==v0 gives
                        # cycle length d+1+1 = d+2? Let's be careful.
                        # dist(v0, u) = d. If w == v0, cycle length = d + 1.
                        best = min(best, d + 1)
                        continue
                    if w in visited:
                        # cycle length = dist(v0,w) + d + 1
                        cyc = visited[w] + d + 1
                        if cyc < best:
                            best = cyc
                    else:
                        visited[w] = d + 1
                        dq.append((w, d + 1))
        return best

    # Full girth verification (sampled if too slow)
    import time
    t0 = time.time()
    for v in range(n):
        if time.time() - t0 > 100:
            break
        sc = shortest_cycle_through(v, adj_list, g)
        if sc < g:
            raise RuntimeError("Girth violation: cycle of length %d < %d" % (sc, g))

    return n, edges
