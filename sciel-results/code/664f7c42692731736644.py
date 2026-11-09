"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
from collections import defaultdict
from heapq import heappush, heappop


def _verify(n, k, g, edges):
    """Return True iff (0) simple (1) k-regular (2) girth >= g."""
    if n <= 0:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    if any(d != k for d in deg):
        return False
    # BFS girth check from every vertex; first visit in BFS = shortest cycle
    for start in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[start] = 0
        q = [start]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in seen.get_neighbors(u) if False else _adj_from_edges(u, n, edges):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                elif par[u] != v and par[v] != u:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _adj_from_edges(u, n, edges):
    # O(n) scan; only used inside the final verify
    out = []
    for a, b in edges:
        if a == u:
            out.append(b)
        elif b == u:
            out.append(a)
    return out


def construct(k, g):
    """Greedy tree-embedding with label-repair local search.

    1. Build a k-ary tree of depth D = g // 2 rooted at an anchor node.
       Number of tree nodes T ~ 1 + k * (k**D - 1)/(k - 1).
    2. Initial assignment: every node maps to a distinct vertex (T vertices).
    3. Iteratively pick a pair (a, b) of tree-nodes with:
           dist_T(a, b) < g and label[a] != label[b]
         and remap label[b] -> label[a].
         This is allowed iff (a) no other tree-node c at dist_T(b, c) < g
         currently holds label[a], and (b) for every tree-edge (u, v)
         incident to b, the neighbour's label is not already label[a]
         (which would create a parallel edge or loop).
       Each such move strictly reduces the number of used labels, so the
       process terminates at a local optimum.
    4. Emit edges from the tree, dedupe, and verify.
    """
    D = g // 2
    # ---------- 1. Build the k-ary tree ----------
    nodes = [0]                       # node 0 = root/anchor
    parent = [-1]
    adj_tree = []

    def new_node(p):
        idx = len(nodes)
        nodes.append(idx)
        parent.append(p)
        adj_tree.append([])
        if p != -1:
            adj_tree[p].append(idx)
        return idx

    adj_tree.append([])
    # BFS grow to depth D
    frontier = [0]
    depth = 0
    while depth < D:
        nxt = []
        for u in frontier:
            for _ in range(k):
                v = new_node(u)
                adj_tree[u].append(v)
                adj_tree[v].append(u)
                nxt.append(v)
        frontier = nxt
        depth += 1
    T = len(nodes)

    # Precompute pair-list of tree-nodes at tree-distance < g,
    # but only keep pairs that can meaningfully be merged (i.e., distinct
    # nodes). We will lazily check the current labels.
    # To keep memory small, for each node store its ball of radius r < g
    # via BFS on the tree (tree -> distance = depth sum).
    # For g = 10, r = 4..9; ball sizes for k=4: 1+4+16+64+256+1024+4096+... 
    #  which grows. For g=10, k=4: r=9 is huge. Instead, we work with
    # a sliding strategy: only consider pairs whose tree-distance is
    # exactly 2 * (D-1) - 1 .. 2*(D-1)  (the most "tight" pairs), plus a
    # random subset. Simpler: for each node, BFS up to radius g-1 but cap
    # the ball to the first B neighbours.
    # We use a randomized greedy that, at each step, samples candidate
    # pairs from the current frontier of "still-distinct-label" nodes.

    # ---------- 2. Initial assignment ----------
    label = list(range(T))            # label[i] = vertex id of tree-node i
    vertex_of = {i: i for i in range(T)}

    # Reverse map: vertex -> tree-node(s) holding it.  Initially 1-to-1.
    owner = list(range(T))            # owner[v] = tree-node holding v

    # For fast "is vertex X used by a tree-node within tree-distance < g of
    # node i?" queries, maintain for each vertex the set of tree-nodes that
    # currently hold it (small; usually 1).
    holders = defaultdict(list)
    for i, v in enumerate(label):
        holders[v].append(i)

    # Tree distances: for k=4, g=10, T ~ 1 + 4 + 16 + 64 + 256 = 341 (D=5)
    # Precompute all-pairs distances? T^2 ~ 116k, fine.
    # Compute dist[i][j] via BFS from each i on the tree (small tree).
    dist = [[0] * T for _ in range(T)]
    for s in range(T):
        d = [-1] * T
        d[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj_tree[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(v)
        dist[s] = d

    # For each vertex, list of tree-nodes at tree-distance < g from it
    # (i.e. nodes that *conflict* with it for the label).
    conflict_ball = []
    for i in range(T):
        conflict_ball.append([j for j in range(T) if j != i and dist[i][j] < g])

    # ---------- 3. Greedy label repair ----------
    # We'll repeatedly:
    #   - pick a "conflict pair" (i, j) with i in conflict_ball[j],
    #     label[i] != label[j]
    #   - try to merge: label[j] := label[i], if legal
    #   - if no merge succeeds, stop
    # Use a randomized order to escape poor local optima; repeat a few
    # times and keep the best (fewest labels).

    def try_merge(i, j, rng):
        """Attempt to set label[j] = label[i]. Return True on success."""
        vi = label[i]
        if vi == label[j]:
            return False
        # (a) no other holder of vi lies in conflict_ball[j]
        for h in holders[vi]:
            if h != i and h != j:
                if dist[j][h] < g:
                    return False
        # (b) for every tree-neighbour m of j, label[m] != vi
        for m in adj_tree[j]:
            if label[m] == vi:
                return False
        # also: label[i] must not equal label of any tree-neighbour of i
        # (else merging would create a parallel edge between i and that
        # neighbour -- both would map to vi).  But that condition already
        # holds because (i, m) is a tree-edge => dist = 1 < g => m is in
        # conflict_ball[i], and label[i] != label[m] is guaranteed by our
        # invariant that no conflict pair has equal labels.  We re-assert:
        for m in adj_tree[i]:
            if label[m] == vi:
                return False
        # Perform merge
        old_v = label[j]
        holders[old_v].remove(j)
        if not holders[old_v]:
            del holders[old_v]
        label[j] = vi
        holders[vi].append(j)
        return True

    def local_search(max_iters=200000, rng=None):
        """Randomized greedy merge loop."""
        if rng is None:
            rng = random.Random(12345)
        # Build a list of candidate (i, j) pairs we might try.
        # i = "donor", j = "receiver"; we try label[j] := label[i].
        # For efficiency, iterate receivers j and pick a random donor i
        # from conflict_ball[j] whose label differs.
        # We track a "blocked" flag per receiver to avoid re-scanning.
        # Simple approach: maintain a heap keyed by (len(conflict_ball[j]), j)
        # so we try receivers with the most conflict partners first.
        import heapq
        heap = [(len(conflict_ball[j]), j) for j in range(T)]
        heapq.heapify(heap)
        iters = 0
        progress = 0
        while iters < max_iters:
            iters += 1
            # Pop a receiver with many conflicts
            _, j = heapq.heappop(heap)
            done = True
            for i in conflict_ball[j]:
                if label[i] != label[j]:
                    if try_merge(i, j, rng):
                        progress += 1
                        # re-push with updated count
                        new_c = 0
                        for h in conflict_ball[j]:
                            if label[h] != label[j]:
                                new_c += 1
                        heapq.heappush(heap, (new_c, j))
                        done = False
                        break
            if done:
                # re-push and continue
                new_c = 0
                for h in conflict_ball[j]:
                    if label[h] != label[j]:
                        new_c += 1
                heapq.heappush(heap, (new_c, j))
            if progress % 200 == 0 and not heap:
                break
            if not heap:
                break
        # Count distinct labels
        return len(set(label))

    # Try a few restarts with different random seeds; keep best.
    best_label = None
    best_n = T
    for seed in range(6):
        rng = random.Random(seed * 7919 + 1)
        # Reset state
        label = list(range(T))
        holders = defaultdict(list)
        for i, v in enumerate(label):
            holders[v].append(i)
        n_used = local_search(max_iters=150000, rng=rng)
        if n_used < best_n:
            best_n = n_used
            best_label = list(label)

    label = best_label

    # ---------- 4. Emit edges ----------
    edges = []
    seen_edges = set()
    for i in range(T):
        for j in adj_tree[i]:
            if i < j:
                u, v = label[i], label[j]
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in seen_edges:
                    continue
                seen_edges.add((a, b))
                edges.append((u, v))

    n = max(label) + 1 if label else 0

    # ---------- 5. Verify (cheap) ----------
    # Rebuild adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Degree check
    if any(len(a) != k for a in adj):
        # Fall back to L-U-W (known valid)
        return _luw_fallback(k, g)
    # Girth check via BFS from every vertex
    for s in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                elif par[u] != v and par[v] != u:
                    if dist[u] + dist[v] + 1 < g:
                        return _luw_fallback(k, g)
    return n, edges


def _luw_fallback(k, g):
    """Original L-U-W construction as a guaranteed-valid fallback."""
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
