import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 112.0
    rng = random.Random(777)

    # ---------- seed fallback (LUW) ----------
    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
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

    def full_girth_check(adj, n, g):
        for start in range(n):
            dist = {}
            queue = [(nb, start, 1) for nb in adj[start]]
            for nb in adj[start]:
                dist[(nb, start)] = 1
            head = 0
            while head < len(queue):
                u, prev_u, d = queue[head]
                head += 1
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev_u:
                        continue
                    if v == start:
                        if d + 1 < g:
                            return False
                    elif (v, u) in dist:
                        if d + 1 < g:
                            return False
                    else:
                        dist[(v, u)] = d + 1
                        queue.append((v, u, d + 1))
        return True

    # ---------- caged tree + leaf matching ----------
    def build_tree(k, D):
        """Build a k-ary tree of depth D. Returns (n, children, parent, depth, leaves)."""
        n = 1
        children = [[]]
        parent = [-1]
        depth = [0]
        for _ in range(D):
            base = len(children)
            for i in range(base):
                new = []
                for _ in range(k):
                    new.append(n)
                    n += 1
                    children.append([])
                    parent.append(i)
                    depth.append(depth[i] + 1)
                children[i] = new
        leaves = [i for i in range(n) if depth[i] == D]
        return n, children, parent, depth, leaves

    def lca_dist(parent, depth, a, b):
        """Length of the path between a and b in the tree (via LCA)."""
        da, db = depth[a], depth[b]
        while da > db:
            a = parent[a]
            da -= 1
        while db > da:
            b = parent[b]
            db -= 1
        while a != b:
            a = parent[a]
            b = parent[b]
        return depth[a] * 2 - (depth[a] + depth[b]) + (depth[a] - depth[a]) + (depth[a] + depth[b] - 2 * depth[a]) + (depth[a] - depth[a])
        # simpler:
        # return (original depths) - 2*depth[lca]
    def lca_dist(parent, depth, a, b):
        da, db = depth[a], depth[b]
        x, y = a, b
        while da > db:
            x = parent[x]; da -= 1
        while db > da:
            y = parent[y]; db -= 1
        while x != y:
            x = parent[x]
            y = parent[y]
        return depth[a] + depth[b] - 2 * depth[x]

    def try_matching(k, D, max_tries=300, time_limit=None):
        n, children, parent, depth, leaves = build_tree(k, D)
        m = len(leaves)  # = k * k^(D-1)
        # Build tree adjacency
        adj = [[] for _ in range(n)]
        for u in range(n):
            for c in children[u]:
                adj[u].append(c)
                adj[c].append(u)

        # Precompute pair distances for the leaf set (m^2/2). For m<=~3000 ok.
        # We only need to test pairs; compute on the fly per candidate to save memory.
        # leaf index map
        leaf_pos = {v: i for i, v in enumerate(leaves)}

        def dist(a, b):
            return lca_dist(parent, depth, a, b)

        def valid_pair(i, j):
            # cycle length = dist + 2 (two leaf edges) must be >= g
            return dist(leaves[i], leaves[j]) + 2 >= g

        best = None
        for _ in range(max_tries):
            if time_limit is not None and time.time() > time_limit:
                break
            perm = list(range(m))
            rng.shuffle(perm)
            ok = True
            # pair consecutive: (perm[0],perm[1]), (perm[2],perm[3]), ...
            for t in range(0, m, 2):
                i, j = perm[t], perm[t + 1]
                if not valid_pair(i, j):
                    ok = False
                    break
            if not ok:
                continue
            # build full graph
            full_adj = [list(adj[u]) for u in range(n)]
            for t in range(0, m, 2):
                u = leaves[perm[t]]
                v = leaves[perm[t + 1]]
                full_adj[u].append(v)
                full_adj[v].append(u)
            # degree check
            for u in range(n):
                if len(full_adj[u]) != k:
                    ok = False
                    break
            if not ok:
                continue
            # full girth check
            if full_girth_check(full_adj, n, g):
                edges = []
                for u in range(n):
                    for v in full_adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        return None

    # ---------- main search ----------
    # depth D of the caged tree. n = 1 + k*(k^D - 1)/(k-1).
    # For k=3: D=6 -> n=364 (below Moore bound, infeasible); D=7 -> 1092; D=8 -> 3277.
    # Try depths whose n is >= Moore bound first, ascending, to minimize n.
    moore = 2 + (k - 1) * (k ** (g // 2 - 1) - 1) // (k - 2)

    D_candidates = []
    D = 1
    while True:
        nD = 1 + k * (k ** D - 1) // (k - 1)
        D_candidates.append((nD, D))
        if nD > 4000:
            break
        D += 1

    # order by n ascending but only those >= moore
    feasible = [(nD, D) for (nD, D) in D_candidates if nD >= moore]
    # If none feasible (all below moore), take the largest few (closest to moore)
    if not feasible:
        feasible = D_candidates[-4:]

    for nD, D in feasible:
        if time.time() > deadline - 20:
            break
        res = try_matching(k, D, max_tries=400, time_limit=deadline - 20)
        if res is not None:
            return res

    # Fallback: seed
    return seed_construct(k, g)
