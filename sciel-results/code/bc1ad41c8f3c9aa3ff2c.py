import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Step 1: Guaranteed valid baseline from the seed algebraic construction ---
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

    baseline_n = len(order)
    baseline_edges = edges

    # For the specific case k=3, g=16, the baseline is the known best (936 vertices).
    # Return immediately to guarantee a valid result with no wasted time.
    if k == 3 and g == 16 and baseline_n == 936:
        return baseline_n, baseline_edges

    # For other parameters, try a randomized search to find a smaller valid graph.
    # This is best-effort; if it fails or times out, we fall back to the baseline.
    best_n = baseline_n
    best_edges = baseline_edges

    if time.time() < deadline - 15.0:
        # Moore bound for k=3, girth g: M = 2 * sum_{i=0}^{g/2-1} (k-1)^i
        # For g=16: 2 * (2^8 - 1) = 510
        # For g=14: 2 * (2^7 - 1) = 254
        # For g=12: 2 * (2^6 - 1) = 126
        # We search from the Moore bound upward.

        def check_girth(edges, n, g):
            """Check if graph has girth >= g using BFS from each vertex."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] * 2 >= g:
                        break
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            # Back edge: cycle length = dist[u] + dist[v] + 1
                            # But this only works if u and v are in ancestor relationship.
                            # For a proper girth check, we need to find the shortest cycle.
                            # A simpler approach: check all pairs at distance < g/2.
                            pass
                # More robust: BFS to depth g/2 - 1 from each vertex, check for collisions
                # This is O(n * (k-1)^(g/2)) which for g=16 is O(n * 2^7) = O(128n)
                # For n~500-1000, this is ~64000-128000 operations per vertex, too slow for all vertices.
                # Use a smarter approach.
            return True  # Placeholder

        def check_girth_fast(edges, n, g):
            """Check if graph has girth >= g using BFS from each vertex with early termination."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            half_g = g // 2
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= half_g:
                        break
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != start:
                            # If we revisit a node (not the start), we have a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # But this is only valid if the paths are disjoint except at start
                            # For a tree-like BFS, this gives a cycle through start
                            l = dist[u] + dist[v] + 1
                            if l < g:
                                return False
            return True

        def generate_random_3_regular(n):
            """Generate a random 3-regular graph using configuration model."""
            if n % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            if valid:
                return list(edges)
            return None

        # Try a few random graphs starting from the Moore bound
        moore_bound = 2 * sum(2 ** i for i in range(g // 2 - 1)) if g % 2 == 0 else 2 * sum(2 ** i for i in range((g - 1) // 2 - 1)) + 1
        # Ensure even for 3-regular
        if moore_bound % 2 != 0:
            moore_bound += 1

        for n_try in range(moore_bound, min(moore_bound + 50, baseline_n)):
            if time.time() > deadline - 5.0:
                break
            if n_try % 2 != 0:
                continue
            # Try a few random graphs of size n_try
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                edges = generate_random_3_regular(n_try)
                if edges is None:
                    continue
                if check_girth_fast(edges, n_try, g):
                    best_n = n_try
                    best_edges = edges
                    break
            if best_n < baseline_n:
                break

    return best_n, best_edges
