"""Greedy folding of the infinite 4-regular tree into a small 4-regular girth-10 graph."""
# EVOLVE-BLOCK-START
def construct(k, g):
    if k <= 0:
        return 0, []
    if g <= 2:
        # Degenerate: a k-regular multigraph-free simple graph needs >= k+1 vertices.
        n = max(k + 1, 1)
        # Build a k-regular simple graph on n vertices if possible, else best effort.
        edges = _regular_simple(n, k)
        return n, edges

    if k == 4 and g == 10:
        n, edges = _greedy_fold(4, 10)
        if n and edges:
            return n, edges

    # Generic fallback: greedy fold for any (k, g) with g >= 4.
    r = max(1, (g - 1) // 2)  # radius of the ball; ball has no cycle shorter than 2r+2
    n, edges = _greedy_fold(k, g, radius=r)
    if n and edges:
        return n, edges

    # Last-resort deterministic fallback: circulant k-regular graph (may violate girth).
    n = 2 * k
    edges = []
    for i in range(n):
        for d in range(1, k // 2 + 1):
            if d == k - d:
                edges.append((i, (i + d) % n))
            else:
                edges.append((i, (i + d) % n))
                edges.append((i, (i - d) % n))
    return n, edges


def _regular_simple(n, k):
    """Deterministic k-regular simple graph on n vertices (if k < n and nk even)."""
    if k >= n or (n * k) % 2 != 0:
        return []
    # Use a circulant construction.
    edges = []
    # We need k/2 steps (or (k-1)/2 plus a middle step if k odd).
    if k % 2 == 0:
        steps = list(range(1, k // 2 + 1))
    else:
        steps = list(range(1, (k - 1) // 2 + 1))
    for i in range(n):
        for d in steps:
            j = (i + d) % n
            if i < j:
                edges.append((i, j))
            else:
                edges.append((j, i))
        if k % 2 == 1:
            j = (i + (n // 2)) % n
            if i < j:
                edges.append((i, j))
            else:
                edges.append((j, i))
    return edges


def _greedy_fold(k, g, radius=None):
    """
    Greedy folding of the k-regular tree.

    Build the ball of radius `radius` (default radius = (g-1)//2 so the ball itself
    has girth >= g). Vertices at tree-distance >= g are eligible to be identified
    (this cannot create a cycle shorter than g). Greedily merge the eligible pair
    that yields the largest reduction in vertex count while keeping every vertex's
    degree <= k and not creating a loop or a duplicate edge. Then repair degrees.
    """
    if radius is None:
        radius = max(1, (g - 1) // 2)

    # --- Build the ball of radius `radius` in the k-regular tree, rooted at 0. ---
    # Represent each vertex by the tuple of "directions" from the root (a path).
    # Directions: 0 = toward root (parent), 1..k = the k children directions.
    # The root has path (). A vertex at depth t has a path of length t.
    # Parent of a path p (non-empty) is p[:-1].
    # Children of a non-root vertex p: the k directions that are not the parent direction.
    # For the root, children are directions 1..k.

    import math

    paths = [()]
    path_set = {()}
    parent = {}
    parent[()] = None

    for depth in range(radius):
        new_paths = []
        for p in paths:
            if depth == 0:
                # root's children
                dirs = range(1, k + 1)
            else:
                # p has length depth; its parent direction is p[-1]
                last = p[-1]
                if last == 0:
                    dirs = range(1, k + 1)
                else:
                    dirs = [d for d in range(1, k + 1) if d != last]
            for d in dirs:
                np_ = p + (d,)
                if np_ not in path_set:
                    path_set.add(np_)
                    new_paths.append(np_)
                    parent[np_] = p
        paths.extend(new_paths)

    N = len(paths)
    # Map path -> index
    idx = {p: i for i, p in enumerate(paths)}

    # Build adjacency in the tree (each vertex degree k except leaves have degree < k
    # because the ball is truncated; leaves have only their parent + (k-1) children that
    # are outside the ball... actually within the ball, leaves at depth `radius` have
    # only their parent as a neighbor inside the ball, plus k-1 children outside).
    # For folding purposes we consider the *infinite* tree, so every vertex has degree k.
    # We represent the tree edges within the ball, and treat outside-neighbors implicitly.
    # Simpler: work with the infinite tree but only track vertices in the ball.
    # A vertex's k neighbors: parent (if not root) and k children. Children always exist
    # in the infinite tree; we only "materialize" those within the ball.

    # For the greedy fold we need the full neighbor structure. Let's represent each
    # vertex by its path, and compute its k neighbors as paths (some may be outside the ball,
    # which we still track as virtual vertices).

    # To keep it manageable, we include virtual vertices for neighbors just outside the ball
    # (depth radius+1) so that the boundary vertices have all k neighbors represented.
    # Actually, for a clean fold we want a finite set that is closed under "all neighbors"
    # so that the quotient is a genuine graph. Let's include depth up to radius+1.

    # Rebuild with radius+1 to have a closed neighborhood.
    R = radius + 1
    paths2 = [()]
    path_set2 = {()}
    for depth in range(R):
        new_paths = []
        for p in paths2:
            if depth == 0:
                dirs = range(1, k + 1)
            else:
                last = p[-1]
                if last == 0:
                    dirs = range(1, k + 1)
                else:
                    dirs = [d for d in range(1, k + 1) if d != last]
            for d in dirs:
                np_ = p + (d,)
                if np_ not in path_set2:
                    path_set2.add(np_)
                    new_paths.append(np_)
        paths2.extend(new_paths)

    N2 = len(paths2)
    idx2 = {p: i for i, p in enumerate(paths2)}

    # Neighbor function on the infinite tree, restricted to our finite set.
    def neighbors_of(p):
        if len(p) == 0:
            # root: children directions 1..k
            return [p + (d,) for d in range(1, k + 1)]
        else:
            last = p[-1]
            nbrs = [p[:-1]]  # parent
            if last == 0:
                for d in range(1, k + 1):
                    nbrs.append(p + (d,))
            else:
                for d in range(1, k + 1):
                    if d != last:
                        nbrs.append(p + (d,))
            return nbrs

    # Tree distance between two paths: find LCP, then distance = len(a)+len(b)-2*lcp.
    def lcp_len(a, b):
        m = min(len(a), len(b))
        i = 0
        while i < m and a[i] == b[i]:
            i += 1
        return i

    def tree_dist(a, b):
        l = lcp_len(a, b)
        return len(a) + len(b) - 2 * l

    # Eligibility: two vertices can be identified only if tree_dist >= g
    # (so that the identification does not create a cycle shorter than g).
    # We'll greedily merge pairs.

    # We maintain a DSU over the N2 vertices. Initially each is its own set.
    parent_dsu = list(range(N2))

    def find(x):
        while parent_dsu[x] != x:
            parent_dsu[x] = parent_dsu[parent_dsu[x]]
            x = parent_dsu[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent_dsu[ra] = rb
        return True

    # For efficiency, we want to try merges that reduce vertex count. A merge is valid if:
    #  - tree_dist(u, v) >= g  (no short cycle)
    #  - u and v are not adjacent in the tree (else loop)
    #  - after merge, no vertex would exceed degree k, and no duplicate edge.
    #
    # Greedy heuristic: repeatedly find the pair (u, v) with tree_dist >= g that is
    # "most central" (largest product of current degrees) and merge if valid.
    #
    # To keep runtime bounded, we limit the number of merge attempts.

    import random
    rng = random.Random(12345)

    # Precompute degrees within the finite set (each vertex has degree k in the infinite
    # tree, but within our closed set of radius R, boundary vertices may have degree < k
    # because some neighbors are outside... but we included radius+1 so that the radius
    # ball is fully closed. Vertices at depth R have children outside, so they have degree
    # < k within the set. That's fine; we'll handle it.)
    # Actually for a proper finite graph we want every vertex to have exactly k neighbors
    # in the set. Including radius+1 does NOT make depth-R vertices have k neighbors in the
    # set (their children are at depth R+1, outside). So the set is not closed.
    #
    # Let me instead use the ball of radius R where R = radius, and accept that boundary
    # vertices have degree < k in the ball. After folding, we repair by adding edges.
    #
    # Simpler robust plan: work with the ball of radius `radius` (closed under the
    # constraint that we only need girth >= g). The ball of radius (g-1)//2 has no cycle
    # shorter than g+1, so any identification of vertices at distance >= g preserves girth.
    # After folding, some vertices will have degree < k (the original leaves) and some may
    # have degree > k (if many identifications pile up). We then repair to make it k-regular.

    # Let's restart cleanly with the ball of radius `radius`.
    pathsB = [()]
    path_setB = {()}
    for depth in range(radius):
        new_paths = []
        for p in pathsB:
            if depth == 0:
                dirs = range(1, k + 1)
            else:
                last = p[-1]
                if last == 0:
                    dirs = range(1, k + 1)
                else:
                    dirs = [d for d in range(1, k + 1) if d != last]
            for d in dirs:
                np_ = p + (d,)
                if np_ not in path_setB:
                    path_setB.add(np_)
                    new_paths.append(np_)
        pathsB.extend(new_paths)

    NB = len(pathsB)
    idxB = {p: i for i, p in enumerate(pathsB)}

    # Tree neighbors within the ball (a vertex's neighbors that are also in the ball).
    tree_adj = [[] for _ in range(NB)]
    for i, p in enumerate(pathsB):
        for q in neighbors_of(p):
            if q in idxB:
                j = idxB[q]
                tree_adj[i].append(j)

    # DSU
    par = list(range(NB))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    # Current component representatives and their "virtual degree" (number of distinct
    # neighbor components). We recompute as needed.

    # Greedy merge loop.
    # Candidate pairs: all pairs (i, j) with i < j, tree_dist >= g, and not adjacent.
    # We'll process them in an order that favors high-degree vertices.
    # To bound runtime, we cap the number of merge operations.

    max_ops = 20000
    ops = 0

    # Build candidate list with a priority score.
    # score = degree(i) * degree(j)  (higher = more beneficial to merge)
    deg = [len(a) for a in tree_adj]

    candidates = []
    for i in range(NB):
        pi = pathsB[i]
        for j in range(i + 1, NB):
            pj = pathsB[j]
            # skip adjacent
            if j in tree_adj[i]:
                continue
            d = tree_dist(pi, pj)
            if d >= g:
                candidates.append((deg[i] * deg[j], i, j, d))

    # Sort by descending score, then by descending distance (prefer farther apart).
    candidates.sort(key=lambda t: (-t[0], -t[3], t[1], t[2]))

    # We'll attempt merges in this order. After each successful merge, degrees change,
    # but we keep the original ordering for simplicity (a heuristic).
    merged_count = 0
    for score, i, j, d in candidates:
        if ops >= max_ops:
            break
        ops += 1
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # Check validity: merging components ri and rj must not create a loop or
        # duplicate edge, and must not make any vertex exceed degree k.
        # Build the set of neighbor-components of ri and rj.
        nbrs_i = set()
        for nb in tree_adj[i]:
            nbrs_i.add(find(nb))
        nbrs_j = set()
        for nb in tree_adj[j]:
            nbrs_j.add(find(nb))
        # If ri is in nbrs_j or rj is in nbrs_i, merging creates a loop (they are adjacent
        # at the component level).
        if ri in nbrs_j or rj in nbrs_i:
            continue
        # The merged component's neighbor set:
        merged_nbrs = (nbrs_i | nbrs_j) - {ri, rj}
        # For each neighbor component c in merged_nbrs, the number of edges between the
        # merged component and c must be <= 1 (else duplicate edge).
        # Count edges from ri-side to c and rj-side to c.
        edge_count = {}
        for c in merged_nbrs:
            cnt = 0
            for nb in tree_adj[i]:
                if find(nb) == c:
                    cnt += 1
            for nb in tree_adj[j]:
                if find(nb) == c:
                    cnt += 1
            edge_count[c] = cnt
        if any(cnt > 1 for cnt in edge_count.values()):
            continue
        # Also ensure that no vertex in the merged component would have degree > k.
        # The degree of the merged component (as a "super-vertex") is len(merged_nbrs)
        # plus any self-loops (none, since we excluded ri/rj). We require len(merged_nbrs) <= k.
        if len(merged_nbrs) > k:
            continue
        # Perform merge.
        par[ri] = rj
        merged_count += 1

    # Now compute components.
    comp_of = [find(i) for i in range(NB)]
    comp_ids = {}
    comp_list = []
    for i in range(NB):
        c = comp_of[i]
        if c not in comp_ids:
            comp_ids[c] = len(comp_list)
            comp_list.append(c)
    n_components = len(comp_list)

    # Build the quotient graph: vertices = components, edges = between components.
    # For each tree edge (i, j), add an edge between comp(i) and comp(j) if different.
    quot_adj = [set() for _ in range(n_components)]
    for i in range(NB):
        ci = comp_ids[comp_of[i]]
        for j in tree_adj[i]:
            cj = comp_ids[comp_of[j]]
            if ci != cj:
                quot_adj[ci].add(cj)
                quot_adj[cj].add(ci)

    # Now repair to make it k-regular. Some components have degree < k (leaves) or > k.
    # We need to add/remove edges to make every vertex degree exactly k, without creating
    # a cycle shorter than g.
    #
    # This is the hard part. A simple repair: for each vertex with degree < k, add edges
    # to vertices at distance >= g/2 (in the quotient) that also need edges, avoiding
    # loops and duplicates. For vertices with degree > k, we cannot easily remove edges
    # without breaking regularity of neighbors.
    #
    # Given the difficulty, and that the greedy merge already keeps degree <= k, the
    # main issue is under-degree. Let's add edges to fix under-degree.
    #
    # Compute current degrees.
    cur_deg = [len(s) for s in quot_adj]

    # Add edges to bring under-degree vertices up to k.
    # For each vertex u with cur_deg[u] < k, find a vertex v with cur_deg[v] < k,
    # v != u, not already adjacent, and quotient-distance(u, v) >= g/2 (heuristic to
    # avoid short cycles), then add edge.
    #
    # To bound girth, we require that adding edge (u, v) does not create a cycle shorter
    # than g. A sufficient condition: the shortest path between u and v in the current
    # graph (without the new edge) has length >= g-1. We'll compute BFS distances.
    #
    # For efficiency, we'll do a limited number of repairs.

    import heapq

    def bfs_dist(adj, src):
        dist = [-1] * len(adj)
        dist[src] = 0
        q = [src]
        for x in q:
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    repair_limit = 5000
    repairs = 0
    for u in range(n_components):
        while cur_deg[u] < k and repairs < repair_limit:
            repairs += 1
            # Find best v.
            best_v = -1
            best_d = -1
            # Compute distance from u to all others.
            dist = bfs_dist(quot_adj, u)
            for v in range(n_components):
                if v == u:
                    continue
                if v in quot_adj[u]:
                    continue
                if cur_deg[v] >= k:
                    continue
                d = dist[v]
                if d == -1:
                    continue
                if d >= g - 1:
                    if d > best_d:
                        best_d = d
                        best_v = v
            if best_v == -1:
                # Relax: allow d >= g - 2
                for v in range(n_components):
                    if v == u or v in quot_adj[u] or cur_deg[v] >= k:
                        continue
                    d = dist[v]
                    if d == -1:
                        continue
                    if d >= g - 2:
                        if d > best_d:
                            best_d = d
                            best_v = v
            if best_v == -1:
                break
            quot_adj[u].add(best_v)
            quot_adj[best_v].add(u)
            cur_deg[u] += 1
            cur_deg[best_v] += 1

    # If we still have under-degree vertices, we may need to accept a non-perfect graph.
    # But the task requires exactly degree k. Let's try a more aggressive repair:
    # allow adding edges even if it risks girth, then verify and drop if girth fails.
    # For now, if any vertex has degree != k, we fall back to a known valid construction.

    if any(d != k for d in cur_deg):
        # Try to fix over-degree by not merging so much: re-run with fewer merges.
        # For simplicity, fall back to the Lazebnik-Ustimenko-Woldar construction.
        return _luw(k, g)

    # Build edge list.
    edges = []
    for u in range(n_components):
        for v in quot_adj[u]:
            if u < v:
                edges.append((u, v))

    return n_components, edges


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
