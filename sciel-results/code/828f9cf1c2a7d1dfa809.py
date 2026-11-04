import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for the 150s limit

    # LUW graph construction (Lucas-type universal covers)
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

    # Build adjacency list from edges
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_degree(adj, k):
        return all(len(a) == k for a in adj)

    # Non-backtracking BFS girth check from a single vertex
    def min_nonbacktrack_cycle_from(adj, s):
        from collections import deque
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                return depth + 1
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
        return float('inf')

    def girth_of_graph(n, adj, k):
        # Quick check: all degrees must be k
        if not check_degree(adj, k):
            return 0
        # Find minimum non-backtracking cycle
        best = float('inf')
        for s in range(n):
            if time.time() > deadline:
                break
            c = min_nonbacktrack_cycle_from(adj, s)
            if c < best:
                best = c
            if best <= 3:
                break
        return best

    # Local search: swap two edges to reduce girth or keep same n
    def local_search(n, adj, k, g, max_iters=500):
        # Build edge list
        edge_set = set()
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
                    edges.append((u, v))
        # For each vertex, store its neighbors in a list
        adj_list = [list(a) for a in adj]

        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            adj_list[u].remove(v)
            adj_list[v].remove(u)

        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            adj_list[u].append(v)
            adj_list[v].append(u)

        best_n = n
        best_adj = None
        best_girth = float('inf')

        # Compute initial girth
        init_girth = girth_of_graph(n, adj, k)
        best_girth = init_girth
        if init_girth >= g:
            best_adj = [set(a) for a in adj]

        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick random edge to remove
            e_idx = np.random.randint(len(edges))
            u, v = edges[e_idx]
            # Pick another random edge
            e2_idx = np.random.randint(len(edges))
            if e2_idx == e_idx:
                continue
            w, x = edges[e2_idx]
            # Try swapping (u,v) and (w,x) with (u,w) and (v,x) or (u,x) and (v,w)
            for perm in [((u, w), (v, x)), ((u, x), (v, w))]:
                a1, a2 = perm[0]
                b1, b2 = perm[1]
                # Check validity
                if a1 == a2 or b1 == b2:
                    continue
                if (a1, a2) in adj[a1] or (b1, b2) in adj[b1]:
                    continue
                if (a1, b1) in edge_set or (a2, b2) in edge_set:
                    continue
                # Remove old edges
                remove_edge(u, v)
                remove_edge(w, x)
                edge_set.discard((min(u, v), max(u, v)))
                edge_set.discard((min(w, x), max(w, x)))
                # Add new edges
                add_edge(a1, a2)
                add_edge(b1, b2)
                edge_set.add((min(a1, a2), max(a1, a2)))
                edge_set.add((min(b1, b2), max(b1, b2)))
                # Check if still regular
                if not check_degree(adj, k):
                    # Revert
                    remove_edge(a1, a2)
                    remove_edge(b1, b2)
                    edge_set.discard((min(a1, a2), max(a1, a2)))
                    edge_set.discard((min(b1, b2), max(b1, b2)))
                    add_edge(u, v)
                    add_edge(w, x)
                    edge_set.add((min(u, v), max(u, v)))
                    edge_set.add((min(w, x), max(w, x)))
                    continue
                # Check girth
                new_girth = girth_of_graph(n, adj, k)
                if new_girth >= g and new_girth > best_girth:
                    best_girth = new_girth
                    best_adj = [set(a) for a in adj]
                    # Keep this change
                    break
                # Revert
                remove_edge(a1, a2)
                remove_edge(b1, b2)
                edge_set.discard((min(a1, a2), max(a1, a2)))
                edge_set.discard((min(b1, b2), max(b1, b2)))
                add_edge(u, v)
                add_edge(w, x)
                edge_set.add((min(u, v), max(u, v)))
                edge_set.add((min(w, x), max(w, x)))
        return best_n, best_adj, best_girth

    # Generate LUW graph
    n_luw, edges_luw = luw_construct(k, g)
    adj_luw = build_adj(n_luw, edges_luw)

    # Check if LUW graph is valid
    if check_degree(adj_luw, k):
        girth_luw = girth_of_graph(n_luw, adj_luw, k)
        if girth_luw >= g:
            # LUW graph is valid, try local search to see if we can improve
            # but for now just return it
            return n_luw, edges_luw

    # If LUW didn't work, try local search from LUW
    if time.time() < deadline:
        best_n, best_adj, best_girth = local_search(n_luw, adj_luw, k, g)
        if best_adj is not None and best_girth >= g:
            # Convert adj to edges
            edges_out = []
            for u in range(best_n):
                for v in best_adj[u]:
                    if u < v:
                        edges_out.append((u, v))
            return best_n, edges_out

    # Fallback: return LUW graph
    return n_luw, edges_luw
