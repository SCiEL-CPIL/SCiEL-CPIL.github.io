import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    def get_luw_fallback(k, g):
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
            nbrs = list(neighbours(vertex))
            for other in nbrs:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
            if vertex[0] == 0:
                for other in nbrs:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def has_short_cycle(n, adj, girth_target):
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= girth_target - 1:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return True, cycle_len, (u, v)
        return False, None, None

    def generate_random_regular(n, degree):
        stubs = []
        for i in range(n):
            stubs.extend([i] * degree)
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, 2 * n * degree // 2, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in seen:
                return None
            seen.add(key)
            edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def repair_graph(n, edges, girth_target, max_iter=5000):
        adj = build_adj(n, edges)
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))

        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                return None, None

            short, cycle_len, (u, v) = has_short_cycle(n, adj, girth_target)
            if not short:
                return n, list(edge_set)

            # Try to find a swap that breaks the cycle
            # We need to find two edges (a,b) and (c,d) such that swapping to (a,c) and (b,d)
            # (or (a,d) and (b,c)) removes the short cycle and doesn't create loops or multi-edges
            # and doesn't create a shorter cycle.

            # Heuristic: pick a random edge (a,b) and a random edge (c,d) from the current graph
            # and try the swap.
            # To be more effective, we should target edges near the cycle.
            # Let's just try random pairs first.

            edge_list = list(edge_set)
            random.shuffle(edge_list)

            # Limit the number of swap attempts per iteration to avoid getting stuck
            for _ in range(20):
                if time.time() > deadline - 10:
                    return None, None

                # Pick two random edges
                e1 = edge_list[random.randint(0, len(edge_list) - 1)]
                e2 = edge_list[random.randint(0, len(edge_list) - 1)]

                a, b = e1
                c, d = e2

                # Try swap 1: (a,c) and (b,d)
                # Check validity
                if a != c and b != d and (a, c) != (c, a) and (b, d) != (d, b):
                    # Ensure no loops
                    if a != c and b != d:
                        # Ensure no multi-edges
                        if (min(a, c), max(a, c)) not in edge_set and (min(b, d), max(b, d)) not in edge_set:
                            # Ensure the new edges don't already exist
                            # Remove old edges
                            edge_set.remove(e1)
                            edge_set.remove(e2)
                            new_e1 = (min(a, c), max(a, c))
                            new_e2 = (min(b, d), max(b, d))

                            # Check if new edges are already in set (shouldn't be, but just in case)
                            if new_e1 not in edge_set and new_e2 not in edge_set:
                                edge_set.add(new_e1)
                                edge_set.add(new_e2)

                                # Rebuild adj
                                adj = build_adj(n, list(edge_set))

                                # Check girth
                                short_new, _, _ = has_short_cycle(n, adj, girth_target)
                                if not short_new:
                                    return n, list(edge_set)

                                # If it didn't help or made it worse, revert
                                edge_set.remove(new_e1)
                                edge_set.remove(new_e2)
                                edge_set.add(e1)
                                edge_set.add(e2)
                                adj = build_adj(n, list(edge_set))
                            else:
                                edge_set.add(e1)
                                edge_set.add(e2)
                                adj = build_adj(n, list(edge_set))

                # Try swap 2: (a,d) and (b,c)
                if a != d and b != c:
                    if (min(a, d), max(a, d)) not in edge_set and (min(b, c), max(b, c)) not in edge_set:
                        edge_set.remove(e1)
                        edge_set.remove(e2)
                        new_e1 = (min(a, d), max(a, d))
                        new_e2 = (min(b, c), max(b, c))

                        if new_e1 not in edge_set and new_e2 not in edge_set:
                            edge_set.add(new_e1)
                            edge_set.add(new_e2)
                            adj = build_adj(n, list(edge_set))

                            short_new, _, _ = has_short_cycle(n, adj, girth_target)
                            if not short_new:
                                return n, list(edge_set)

                            edge_set.remove(new_e1)
                            edge_set.remove(new_e2)
                            edge_set.add(e1)
                            edge_set.add(e2)
                            adj = build_adj(n, list(edge_set))
                        else:
                            edge_set.add(e1)
                            edge_set.add(e2)
                            adj = build_adj(n, list(edge_set))

        return None, None

    if k == 4 and g == 10:
        # Try to find a graph with n < 320
        # Known lower bound is 243.
        # Let's try a few values.
        # The repair heuristic is slow, so we can't do many restarts.
        # Let's try N = 250, 260, 270, 280, 290, 300, 310, 320.
        # Actually, random 4-regular graphs of small n often have many short cycles.
        # The repair heuristic might not be effective enough to break all of them.
        # But let's try.

        # Start with 320 to ensure we have a valid fallback quickly if search fails.
        # Actually, the LUW fallback is instant. So we can spend time searching.

        # Let's try a few N values.
        # We'll allocate time per N.
        n_values = [250, 260, 270, 280, 290, 300, 310]

        for N in n_values:
            if time.time() > deadline - 20:
                break

            # Try a few restarts for this N
            for restart in range(5):
                if time.time() > deadline - 20:
                    break

                edges = generate_random_regular(N, 4)
                if edges is None:
                    continue

                n_result, edges_result = repair_graph(N, edges, 10, max_iter=1000)
                if n_result is not None:
                    return n_result, edges_result

        # Fallback to known best
        n, edges = get_luw_fallback(k, g)
        return n, edges

    else:
        n, edges = get_luw_fallback(k, g)
        return n, edges
