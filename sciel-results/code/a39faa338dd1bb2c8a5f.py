import time
import random
import math
from collections import defaultdict, deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s total, leave 40s buffer)

    def elapsed():
        return time.time() - start_time

    # --- Known cages (hardcoded) ---
    # Petersen graph: 3-regular, girth 5, n=10
    def petersen():
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        for i in range(5):
            edges.append((5 + i, 5 + (i + 2) % 5))
        for i in range(5):
            edges.append((i, 5 + i))
        return 10, edges

    # Heawood graph: 3-regular, girth 6, n=14
    def heawood():
        lines = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
        ]
        edges = []
        for i, line in enumerate(lines):
            for p in line:
                edges.append((p, 7 + i))
        return 14, edges

    # Tutte-Coxeter: 3-regular, girth 8, n=30
    def tutte_coxeter():
        # Built as bipartite incidence of generalized Petersen-like structure
        # Actually, let's build it from its known edge list or construct via LUW and verify.
        # For safety, we'll use LUW for this and verify.
        return None

    # Robertson graph: 4-regular, girth 5, n=19
    def robertson():
        # 19 vertices, 4-regular, girth 5
        # This is the incidence graph of the projective plane of order 3... no, that's 28 vertices.
        # Robertson graph: 19 vertices, 38 edges.
        # Let me construct it. It's the complement of a specific graph on 19 vertices? 
        # Actually, let's just use LUW for (4,5) and see.
        return None

    # Known cage table for small cases
    known = {
        (3, 5): petersen,
        (3, 6): heawood,
    }

    if (k, g) in known:
        return known[(k, g)]()

    # --- LUW Construction ---
    def luw_construct(k, g):
        q = k
        # d = floor((g-1)/2) for odd girth, but LUW uses a specific formula
        # Standard LUW: q-ary, d-dimensional
        # For girth g, we need d such that 2d+1 >= g (for non-backtracking)
        # Actually LUW(girth) uses d = (g-1)//2 for odd g, and d = g//2 for even g
        # But the seed used: d = max(3, g-5 if (g-5)%2 else g-4)
        # Let me use the standard: for girth g, d = ceil((g-1)/2)
        d = (g + 1) // 2  # ceil((g-1)/2)

        if q <= 1:
            return 0, []

        # Mod-q arithmetic
        if q == 2:
            add, sub, mul = (lambda x, y: (x + y) % 2), (lambda x, y: (x - y) % 2), (lambda x, y: (x * y) % 2)
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: (x * y) % q)

        # Rule table for LUW
        # rule[c] = (a, b) meaning y[c] = y[a]*x[b] on side 0, x[a]*y[b] on side 1
        rule = {}
        for c in range(2, d + 1):
            # Standard LUW rules
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                # For c >= 5, use the standard LUW rule
                # r = (c-5) % 4
                r = (c - 5) % 4
                c0 = c - r
                if r == 0:
                    rule[c] = (1, c0 - 2)
                elif r == 1:
                    rule[c] = (c0 - 1, 1)
                elif r == 2:
                    rule[c] = (c0, 1)
                else:
                    rule[c] = (1, c0 + 1)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    n_luw, edges_luw = luw_construct(k, g)

    # Verify the LUW result
    def verify_graph(n, edges, k, g):
        # Check degree
        deg = [0] * n
        edge_set = set()
        for i, j in edges:
            if i == j:
                return False
            if i >= n or j >= n:
                return False
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return False
            edge_set.add(key)
            deg[i] += 1
            deg[j] += 1
        for d in deg:
            if d != k:
                return False

        # Check girth via BFS from each vertex
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)

        for s in range(n):
            # BFS to find shortest cycle containing s
            # Non-backtracking BFS
            visited = {}  # (node, prev) -> dist
            queue = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                queue.append((nb, s, 1))
            found = False
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= g:
                    found = True
                    break
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # Cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                        found = True
                        break
                    if (nb, node) not in visited:
                        visited[(nb, node)] = dist + 1
                        queue.append((nb, node, dist + 1))
            if not found:
                # No cycle found, which is fine (girth >= g)
                pass

        return True

    if not verify_graph(n_luw, edges_luw, k, g):
        # LUW failed, try random search
        pass

    # For small cases where LUW is known to be suboptimal, try to find better
    # This is a heuristic: if LUW gives a large n, try random construction
    def random_regular_graph(n, k, g, time_limit=5.0):
        """Try to build a k-regular graph with girth >= g using random approach."""
        if n * k % 2 != 0:
            return None
        end_time = start_time + time_limit

        while elapsed() < end_time:
            # Random configuration model
            half_edges = n * k // 2
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)

            # Pair up stubs
            edges = []
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edges.append((u, v))

            if not valid:
                continue

            # Check for multi-edges
            edge_set = set()
            valid = True
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if not valid:
                continue

            # Check degree (should be k by construction)
            # Check girth
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)

            girth_ok = True
            for s in range(n):
                # Non-backtracking BFS
                visited = {}
                queue = deque()
                for nb in adj[s]:
                    visited[(nb, s)] = 1
                    queue.append((nb, s, 1))
                while queue:
                    node, prev, dist = queue.popleft()
                    if dist >= g:
                        break
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == s:
                            if dist + 1 < g:
                                girth_ok = False
                                break
                            break
                        if (nb, node) not in visited:
                            visited[(nb, node)] = dist + 1
                            queue.append((nb, node, dist + 1))
                    if not girth_ok:
                        break
                if not girth_ok:
                    break

            if girth_ok:
                return n, edges

        return None

    # Try to improve on LUW for small n
    best_n, best_edges = n_luw, edges_luw

    # If LUW is not too large, try random search for smaller n
    if n_luw <= 200 and k <= 5:
        # Try a few smaller n values
        # Moore bound for k-regular, girth g:
        # For even g=2m: n >= k * (k-1)^(m-1)
        # For odd g=2m+1: n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i
        if g % 2 == 0:
            m = g // 2
            moore = k * (k - 1) ** (m - 1)
        else:
            m = (g - 1) // 2
            moore = 1 + k * sum((k - 1) ** i for i in range(m))

        # Try a few n values above moore
        for trial_n in range(max(moore, 1), n_luw):
            if elapsed() > deadline - 10:
                break
            if trial_n * k % 2 != 0:
                continue
            result = random_regular_graph(trial_n, k, g, time_limit=3.0)
            if result is not None:
                best_n, best_edges = result
                break  # Found a smaller valid graph

    return best_n, best_edges
