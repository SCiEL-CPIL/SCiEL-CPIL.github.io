import time
import math
import random
import sys
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # ---------- Full girth verification via non-backtracking BFS (early exit) ----------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = {}
            for nb in adj[s]:
                dist[(nb, s)] = 1
            queue = deque([(nb, s) for nb in adj[s]])
            while queue:
                if time.time() > time_limit:
                    return -1
                node, prev = queue.popleft()
                d = dist[(node, prev)]
                if d >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        cycle_len = d + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        state = (nb, node)
                        if state not in dist:
                            dist[state] = d + 1
                            queue.append((nb, node))
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------- Fallback: deterministic LUW construction (guaranteed valid, larger) ----------
    def _luw_ops(q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            return (lambda x,y: x ^ y), (lambda x,y: x ^ y), (lambda x,y: table[x][y])
        return ((lambda x,y: (x+y) % q), (lambda x,y: (x-y) % q), (lambda x,y: x*y % q))

    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = _luw_ops(q)
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
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

    # ---------- DIRECT BACKTRACKING: build depth-(g/2) tree, close boundary via SAT ----------
    def backtracking_construct():
        if k != 3 or g != 16:
            return None

        half = g // 2  # 7
        # Build the non-backtracking tree of depth `half` from root 0.
        # Node ids assigned in BFS order. Each tree node (except root) has a parent.
        # We record for each node its parent and the "previous" node in the walk (to avoid backtracking).
        # Root has 3 children. Each non-leaf non-root node has 2 children (3-1 for the parent edge).
        # Leaves are at depth `half`; there are 3 * 2^(half-1) = 96 leaves.
        # Each leaf must be connected to exactly 1 boundary edge (to reach degree 3).
        # Interior non-root nodes: degree from tree edges = 2 (parent + child... wait).

        # Let's lay out the tree precisely.
        # depth 0: root (1 node)
        # depth 1: 3 nodes (each child of root)
        # depth 2: each depth-1 node has 2 children (excluding back-edge to root) => 6 nodes
        # depth 3: 12, depth 4: 24, depth 5: 48, depth 6: 96, depth 7: 192 nodes.
        # Total = 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382 nodes.
        #
        # Degrees within the tree:
        #   root: 3 (to its 3 children) -> already degree 3, no boundary edge.
        #   depth 1..6 non-root nodes: have 1 parent + 2 children = 3 tree edges -> degree 3, no boundary edge.
        #   depth 7 (leaves): have 1 parent only = 1 tree edge -> need 2 more edges.
        #
        # Wait, that gives leaves degree 1, needing 2 edges each -> 192*2/2 = 192 boundary edges.
        # But that's too many. Let me reconsider: for a 3-regular graph, the Moore bound tree of depth 7
        # has leaves that need exactly 1 edge each IF the interior nodes have degree 3 within the tree.
        # Re-examine: a node at depth d (1<=d<=6) has 1 parent and 2 children = 3 edges. Good, degree 3.
        # A leaf at depth 7 has 1 parent = 1 edge, needs 2 more. That's 2 edges per leaf.
        # Hmm, that doesn't match the standard cage picture. Let me recompute the Moore bound.
        #
        # Moore bound for degree 3, girth 16 (even): 2 * (3 * (2^7 - 1)/(2-1)) / 2 ... 
        # Standard: for even girth g=2m, n >= 2 * (k-1) * ( (k-1)^{m-1} - 1 ) / (k-2) ... 
        # For k=3, m=8: n >= 2 * 2 * (2^7 - 1) / 1 = 4 * 127 = 508? 
        # Actually the known Moore bound for (3,16) is 512. Let me just use the tree picture:
        # The "tree of radius 7" centered at an edge has 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1)=510 nodes,
        # plus the central edge structure. The point: leaves at depth 7 from an edge-center.
        #
        # For a vertex-centered BFS tree of depth 7 in a 3-regular girth-16 graph:
        #   level 0: 1 (root)
        #   level 1: 3
        #   level 2: 3*2 = 6
        #   level 3: 12, level 4: 24, level 5: 48, level 6: 96, level 7: 192
        #   total = 382. The 192 level-7 nodes each have degree 1 in the tree (only their parent),
        #   so each needs 2 additional edges to reach degree 3. That's 384 half-edges = 192 edges.
        #   But wait, that would mean n is much larger. The issue: in a real girth-16 graph, the
        #   level-7 nodes are NOT all distinct and NOT all leaves needing 2 edges; the tree closes.
        #
        # The correct model: we build the tree of 382 nodes. The 192 leaves each currently have
        # degree 1. To make the graph 3-regular we must add edges so every leaf reaches degree 3,
        # i.e., add 2 edges incident to each leaf (384 leaf-endpoints). These pair up among leaves
        # (and could in principle go to non-leaves, but non-leaves are already degree 3). So all
        # new edges are leaf-leaf. 192 leaves * 2 = 384 endpoints -> 192 new edges, each connecting
        # 2 leaves. But each leaf needs exactly 2 new edges, so each leaf has degree 2 in the
        # "boundary graph" -> the boundary graph on 192 leaves is 2-regular (a union of cycles).
        # Girth >= 16 requires these boundary cycles (combined with tree paths) to be long.
        #
        # This is the classic cage-construction view. We model the boundary as a 2-regular graph
        # (perfect matching twice = 2-regular) on the 192 leaves and search for one with girth 16.
        #
        # Simpler and more robust: model each leaf as needing 2 partners, build a 2-regular
        # multigraph-free graph on leaves, and use backtracking with girth pruning.

        # Build tree
        parent = [-1] * 400
        depth = [0] * 400
        children = [[] for _ in range(400)]
        # Assign ids
        # root = 0
        nodes = [0]
        depth[0] = 0
        parent[0] = -1
        # BFS to assign
        for d in range(0, half):
            for u in list(nodes):
                if depth[u] != d:
                    continue
                n_children = 3 if d == 0 else 2
                for _ in range(n_children):
                    v = len(nodes)
                    nodes.append(v)
                    depth[v] = d + 1
                    parent[v] = u
                    children[u].append(v)

        n_tree = len(nodes)
        leaves = [v for v in range(n_tree) if depth[v] == half]
        n_leaves = len(leaves)  # expect 192
        # Map leaf to compact index
        leaf_id = {v: i for i, v in enumerate(leaves)}

        # Precompute, for each pair of leaves (a,b), the length of the shortest non-backtracking
        # walk from a to b through the tree (which is depth[a]+depth[b]-2*depth[lca], but since
        # both at depth `half`, and tree is a tree, the unique path length = 2*half - 2*depth[lca]).
        # A cycle formed by tree-path(a,b) + boundary edge (a,b) has length = pathlen(a,b) + 1.
        # Also two boundary edges can form longer cycles. For girth >= 16 we need every cycle >= 16.
        # The simplest necessary condition: for every boundary edge (a,b), pathlen(a,b) + 1 >= 16,
        # i.e., pathlen(a,b) >= 15. Since max pathlen between two depth-7 leaves is 14 (if LCA is root,
        # path = 7+7=14), pathlen+1 <= 15 < 16. So NO leaf-leaf edge can be added directly!
        #
        # This means the boundary cannot be a simple leaf-leaf 2-regular graph on this tree.
        # The real cage is NOT a tree-of-depth-7 with leaf-leaf edges; the BFS tree of radius 7
        # from a vertex in a (3,16)-cage does NOT have all leaves distinct with degree 1.
        # In fact, for girth 16, the ball of radius 7 is a tree (no cycles within radius 7 from a
        # vertex... actually girth 16 means no cycle of length < 16, so a cycle of length 16 is
        # allowed and its vertices are within distance 8 of each other). The ball of radius 7
        # around a vertex IS a tree (any two vertices in it have a unique path, and a cycle would
        # need length >= 16 which exceeds 2*7=14... a cycle through the root of length 16 would have
        # vertices at distance up to 8). So ball of radius 7 is a tree of 382 nodes, and the 192
        # leaves each have 2 "dangling" edges that go to vertices OUTSIDE the ball (at distance 8+).
        #
        # So the correct model: the 382 tree nodes are fixed; the 192 leaves each have 2 stubs that
        # must connect to OTHER vertices (the "outside"). The outside has n - 382 vertices. Total n
        # must be at least 512. The outside vertices plus stubs form the rest. This is a much harder
        # global problem and not cleanly a leaf-leaf matching.
        #
        # Given the complexity, the leaf-leaf 2-regular model is INVALID for girth 16 (as shown).
        # Therefore direct backtracking on this tree is not a clean path. I will instead implement a
        # randomized "stitching" search: start from the LUW/Ramanujan graph and do local edge swaps
        # (2-opt / 4-opt on edges) to reduce n is impossible (n fixed by construction), so instead
        # I'll do a randomized search over voltage assignments on a SMALLER base using a different
        # base graph than Petersen, plus a local repair, which is the approach that actually scored
        # ~530-600. But the directive wants a structural change.
        #
        # I'll implement a genuine direct search: random 3-regular graphs of size n (512..700) via
        # configuration model, then prune/repair to enforce girth 16 by edge swaps, checking girth
        # with early exit. This is a fundamentally different family (random + local search) than
        # algebraic lifts.

        return None  # not used; see random_search below

    # ---------- RANDOM 3-REGULAR + LOCAL SEARCH FOR GIRTH ----------
    def random_search_construct():
        if k != 3 or g != 16:
            return None

        def is_valid_3reg(n, edges):
            deg = [0] * n
            eset = set()
            for u, v in edges:
                if u == v:
                    return False
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in eset:
                    return False
                eset.add((a, b))
                deg[u] += 1
                deg[v] += 1
            for d in deg:
                if d != 3:
                    return False
            return True

        def random_3reg(n):
            # Configuration model
            if n % 2 == 0:
                pass
            # need n*3 even -> n even
            if n % 2 != 0:
                return None
            stubs = []
            for v in range(n):
                for _ in range(3):
                    stubs.append(v)
            random.shuffle(stubs)
            edges = []
            ok = True
            for i in range(0, len(stubs), 2):
                a, b = stubs[i], stubs[i+1]
                if a == b:
                    ok = False
                    break
                edges.append((a, b))
            if not ok:
                return None
            if not is_valid_3reg(n, edges):
                return None
            return edges

        def girth_check(n, edges, g, time_limit):
            return check_girth(n, edges, g, time_limit)

        # Search over n from Moore bound upward; for each, try random graphs + local repair.
        # Local repair: if a short cycle is found, pick an edge in the cycle and try swapping
        # it with a random non-edge to break the cycle while keeping 3-regular.
        best = None
        for n in range(512, 720, 4):
            if remaining() < 20:
                break
            attempts = 40
            for att in range(attempts):
                if remaining() < 15:
                    break
                edges = random_3reg(n)
                if edges is None:
                    continue
                # Quick girth check; if passes, done.
                gc = girth_check(n, edges, g, deadline - 12)
                if gc >= g:
                    return (n, edges)
                # Local repair: try to fix short cycles
                for step in range(200):
                    if remaining() < 12:
                        break
                    # find a short cycle
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    short = None
                    for s in range(n):
                        if remaining() < 12:
                            break
                        # BFS to find any cycle < g
                        dist = {s: 0}
                        q = deque([s])
                        found = False
                        while q and not found:
                            u = q.popleft()
                            for w in adj[u]:
                                if w not in dist:
                                    dist[w] = dist[u] + 1
                                    if dist[w] + 1 < g:
                                        # potential; do a proper non-backtracking search
                                        pass
                                    q.append(w)
                        # Use check_girth for a single source is expensive; instead just re-run full
                        # check with early exit.
                        break
                    # Just re-evaluate full girth and if still short, do a random edge swap.
                    # Pick a random edge and a random non-edge, swap if keeps 3-regular & simple.
                    ei = random.randrange(len(edges))
                    u1, v1 = edges[ei]
                    # find a random vertex pair (a,b) not currently an edge, a!=b, and not equal to
                    # u1/v1 in a way that breaks simplicity. We'll pick a random a and b.
                    a = random.randrange(n)
                    b = random.randrange(n)
                    if a == b:
                        continue
                    # ensure (a,b) is not an existing edge and not (u1,v1)
                    eset = set()
                    for x, y in edges:
                        aa, bb = (x, y) if x < y else (y, x)
                        eset.add((aa, bb))
                    ab = (a, b) if a < b else (b, a)
                    uv = (u1, v1) if u1 < v1 else (v1, u1)
                    if ab in eset or ab == uv:
                        continue
                    # swap: remove (u1,v1), add (a,b). This keeps degrees only if {a,b} == {u1,v1}
                    # in multiset sense, which is false. So a single edge swap breaks 3-regularity.
                    # Need a 2-edge swap (4-cycle flip): remove (u1,v1),(u2,v2), add (u1,u2),(v1,v2)
                    # or (u1,v2),(v1,u2). This preserves degrees.
                    # Pick a second edge (u2,v2) random, distinct, and such that new edges are valid.
                    ej = random.randrange(len(edges))
                    if ej == ei:
                        continue
                    u2, v2 = edges[ej]
                    for (na, nb) in [(u1, u2), (v1, v2), (u1, v2), (v1, u2)]:
                        if na == nb:
                            continue
                        ab2 = (na, nb) if na < nb else (nb, na)
                        if ab2 in eset:
                            continue
                        # also need the other new edge
                        # case (u1,u2)&(v1,v2)
                        break
                    # perform flip removing ei, ej adding (u1,u2),(v1,v2)
                    new_edges = [e for i, e in enumerate(edges) if i != ei and i != ej]
                    na, nb = u1, u2
                    if na == nb:
                        continue
                    ab2 = (na, nb) if na < nb else (nb, na)
                    if ab2 in eset:
                        continue
                    nc, nd = v1, v2
                    if nc == nd:
                        continue
                    cd = (nc, nd) if nc < nd else (nd, nc)
                    if cd in eset:
                        continue
                    new_edges.append((na, nb))
                    new_edges.append((nc, nd))
                    if not is_valid_3reg(n, new_edges):
                        continue
                    edges = new_edges
                    gc = girth_check(n, edges, g, deadline - 12)
                    if gc >= g:
                        return (n, edges)
        return best

    # ---------- Try direct random+local search first (structural change) ----------
    result = random_search_construct()
    if result is not None:
        return result

    # ---------- Fallback to guaranteed LUW ----------
    return luw_construct()
