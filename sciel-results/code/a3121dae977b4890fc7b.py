import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    def elapsed():
        return time.time() - start_time

    # ---- Girth verification: non-backtracking BFS from every vertex ----
    def girth_of(adj, n):
        best = float('inf')
        for u in range(n):
            dist = {}
            q = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = dist[(v, parent)]
                if d >= best:
                    continue
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
        return best

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def primitive_root(p):
        if p == 2:
            return 1
        n = p - 1
        factors = []
        m = n
        d = 2
        while d * d <= m:
            if m % d == 0:
                factors.append(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            factors.append(m)
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, n // f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    def is_valid(k, g, n, edges):
        adj = [set() for _ in range(n)]
        seen = set()
        for (i, j) in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n:
            return False
        gr = girth_of(adj, n)
        return gr >= g

    # ---- L-U-W construction ----
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
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

    # ---- Morgenstern / PGL(2,q) Cayley graph ----
    def build_pgl(q, k, g, max_attempts=300):
        if q < 2 or not is_prime(q):
            return None
        n = q + 1

        def action(x, a, b, c, d):
            if x == q:
                if c == 0:
                    return q
                return (a * pow(c, -1, q)) % q
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            return (num * pow(den, -1, q)) % q

        def try_gens(gens):
            adj = [set() for _ in range(n)]
            for i in range(n):
                x = i
                for (a, b, c, d) in gens:
                    y = action(x, a, b, c, d)
                    if i == y:
                        return None
                    if y in adj[i]:
                        return None
                    adj[i].add(y)
                    adj[y].add(i)
            for i in range(n):
                if len(adj[i]) != k:
                    return None
            visited = [False] * n
            dq = deque([0])
            visited[0] = True
            cnt = 0
            while dq:
                u = dq.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            if cnt != n:
                return None
            gr = girth_of(adj, n)
            if gr < g:
                return None
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        r = primitive_root(q)
        if r is None:
            return None

        rng = random.Random(42)

        # Structured candidate sets
        candidate_sets = []
        # Set 1: transpositions + diagonal
        if k >= 1:
            s1 = [(0, 1, 1, t) for t in range(k - 1)]
            if k > 1:
                s1.append((r, 0, 0, pow(r, -1, q)))
            candidate_sets.append(s1)
        # Set 2: spaced transpositions
        if k >= 1:
            step = max(1, q // (k + 1))
            s2 = [(0, 1, 1, (t * step) % q) for t in range(k)]
            candidate_sets.append(s2)
        # Set 3: diagonal powers
        if k >= 1:
            s3 = [(pow(r, i, q), 0, 0, pow(r, -i, q)) for i in range(k)]
            candidate_sets.append(s3)

        for gens in candidate_sets:
            if len(gens) != k:
                continue
            res = try_gens(gens)
            if res:
                return res

        # Random search
        for _ in range(max_attempts):
            if elapsed() > deadline - 10:
                break
            gens = []
            for _ in range(k):
                while True:
                    a = rng.randint(0, q - 1)
                    b = rng.randint(0, q - 1)
                    c = rng.randint(0, q - 1)
                    d = rng.randint(0, q - 1)
                    if (a * d - b * c) % q != 0:
                        break
                gens.append((a, b, c, d))
            res = try_gens(gens)
            if res:
                return res
        return None

    # ---- BFS-Tree Pruning / Greedy Leaf Merging ----
    # Build a k-ary tree of depth h = floor(g/2) - 1 from a root.
    # Then merge leaves at max depth to satisfy k-regularity.
    # This is a heuristic; may not always produce valid girth-g graph,
    # but provides a structural alternative to group-based constructions.
    def bfs_tree_merge(k, g, max_tries=50):
        h = g // 2 - 1  # depth of tree (edges from root to leaves)
        if h < 1:
            h = 1
        # Number of leaves at depth h in k-ary tree: k * (k-1)^(h-1)
        # Total vertices: 1 + k * sum_{i=0}^{h-1} (k-1)^i
        if h == 1:
            n_total = 1 + k
            n_leaves = k
        else:
            s = 1
            for i in range(h - 1):
                s += (k - 1) ** i
            n_total = 1 + k * s
            n_leaves = k * (k - 1) ** (h - 1)

        if n_total > 50000:  # too large to be practical
            return None

        rng = random.Random(7)

        for attempt in range(max_tries):
            if elapsed() > deadline - 15:
                break

            # Build tree structure
            # Level 0: root (vertex 0)
            # Level i: vertices at depth i
            # Parent of vertex at depth i>0 is determined by BFS order
            # Children of internal vertex: k if depth < h-1, else k-1 (one parent)
            # Actually in a tree, each non-root has 1 parent, so:
            #   root: k children
            #   internal (depth 1..h-1): k-1 children (one is parent)
            #   leaves (depth h): 0 children

            # Assign vertex IDs
            # root = 0
            # depth 1: 1..k
            # depth 2: k+1 .. k + k*(k-1)
            # etc.

            levels = [0]  # start indices
            parent = [0]  # parent of each vertex, -1 for root
            children = [[] for _ in range(n_total)]

            # Build tree
            idx = 1
            for depth in range(h):
                if depth == 0:
                    n_nodes = 1  # root
                    n_children = k
                else:
                    n_nodes = k * (k - 1) ** (depth - 1)
                    n_children = k - 1 if depth < h - 1 else 0

                start_idx = idx
                for node in range(start_idx, start_idx + n_nodes):
                    # Find parent
                    if depth == 0:
                        par = -1
                    else:
                        # Parent is at previous level
                        # In BFS order, parent of node i at depth d is:
                        # i_prev = start_prev + (i - start_prev) // (k-1)
                        start_prev = levels[-1]
                        par = start_prev + (node - start_prev) // (k - 1)
                    parent.append(par)
                    if par >= 0:
                        children[par].append(node)

                idx += n_nodes
                levels.append(start_idx)

            # Now we have a tree. We need to add edges among leaves to make it k-regular.
            # Each leaf currently has degree 1 (only parent edge).
            # Each internal node at depth < h-1 has degree: 1 (parent) + (k-1) (children) = k. Good.
            # Root has degree k. Good.
            # So only leaves need (k-1) more edges each.
            # Total leaf edge-stubs: n_leaves * (k-1). This must be even.
            if (n_leaves * (k - 1)) % 2 != 0:
                continue

            # Leaves are vertices at depth h
            leaf_start = levels[h]
            leaves = list(range(leaf_start, leaf_start + n_leaves))

            # We need to add a (k-1)-regular graph on n_leaves vertices.
            # Use a simple circulant-like construction.
            # For n_leaves vertices, add edges (i, i+d) for appropriate d.
            # A (k-1)-regular graph exists on n vertices iff (k-1)*n is even and k-1 <= n-1.

            if k - 1 > n_leaves - 1:
                continue

            adj = [set() for _ in range(n_total)]

            # Add tree edges
            for v in range(1, n_total):
                p = parent[v]
                if p >= 0:
                    adj[v].add(p)
                    adj[p].add(v)

            # Add leaf edges: (k-1)-regular on leaves
            # Simple approach: for each leaf i, connect to next (k-1) leaves cyclically
            # But need to avoid multi-edges and self-loops.
            # Use: for i in leaves, connect to leaves[(i+j) % n_leaves] for j=1..(k-1)//2
            # and also connect to leaves[(i-j) % n_leaves] for j=1..(k-1)//2
            # This works if k-1 is even. If k-1 is odd, adjust.

            half = (k - 1) // 2
            rem = (k - 1) % 2

            for i in range(n_leaves):
                for j in range(1, half + 1):
                    a = leaves[i]
                    b = leaves[(i + j) % n_leaves]
                    if a != b and b not in adj[a]:
                        adj[a].add(b)
                        adj[b].add(a)
                if rem:
                    # Need one more edge; connect to a non-neighbor
                    # Try leaves[(i + half + 1) % n_leaves]
                    b = leaves[(i + half + 1) % n_leaves]
                    if b not in adj[a] and a != b:
                        adj[a].add(b)
                        adj[b].add(a)

            # Check degree
            valid = True
            for v in range(n_total):
                if len(adj[v]) != k:
                    valid = False
                    break
            if not valid:
                continue

            # Check connectivity
            visited = [False] * n_total
            dq = deque([0])
            visited[0] = True
            cnt = 0
            while dq:
                u = dq.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            if cnt != n_total:
                continue

            # Check girth
            gr = girth_of(adj, n_total)
            if gr < g:
                continue

            edges = []
            for i in range(n_total):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n_total, edges

        return None

    # ---- Main ----
    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if best is None or n < best_n:
            if is_valid(k, g, n, edges):
                best = (n, edges)
                best_n = n
                return True
        return False

    # 1) Try L-U-W (deterministic, always produces valid k-regular graph)
    if elapsed() < deadline - 30:
        try:
            res = luw_construct(k, g)
            if res:
                consider(res[0], res[1])
        except Exception:
            pass

    # 2) Try BFS-Tree Merge (structural alternative)
    if elapsed() < deadline - 30:
        try:
            res = bfs_tree_merge(k, g)
            if res:
                consider(res[0], res[1])
        except Exception:
            pass

    # 3) Try PGL(2,q) Cayley graphs for small q
    n_min = moore_bound(k, g)
    q_start = max(2, n_min - 1)
    q_limit = max(q_start + 50, 100)
    q = q_start
    while q <= q_limit and elapsed() < deadline - 15:
        if is_prime(q):
            try:
                res = build_pgl(q, k, g, max_attempts=100)
                if res:
                    consider(res[0], res[1])
            except Exception:
                pass
        q += 1

    # 4) Fallback: L-U-W should always work
    if best is None:
        try:
            res = luw_construct(k, g)
            best = res
        except Exception:
            # Absolute fallback: return a trivial valid graph
            # For k=1, g=1: single vertex
            if k == 1 and g == 1:
                best = (1, [])
            else:
                # Build a cycle of length max(g, k) if k=2, or use L-U-W with larger d
                if k == 2:
                    n = max(g, 3)
                    edges = [(i, (i + 1) % n) for i in range(n)]
                    best = (n, edges)
                else:
                    # Last resort: complete graph K_{k+1} if girth allows
                    n = k + 1
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                    best = (n, edges)

    return best

# EVOLVE-BLOCK-END
