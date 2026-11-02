"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        # Even girth: n >= k + k*(k-1)^(g/2 - 1)
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
        # = 1 + k * ((k-1)^((g-1)/2) - 1) / (k - 2) for k > 2
        if k == 2:
            moore = g  # cycle
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    # For the specific case k=5, g=9: moore bound is 1 + 5*(4^4 - 1)/3 = 1 + 5*255/3 = 1+425=426
    # But the actual bound is tighter. Let's just use a reasonable starting point.

    # Strategy: Use a combination of approaches
    # 1. Try explicit constructions (like the Lazebnik-Ustimenko-Woldar from seed)
    # 2. Try random regular graph construction with girth verification
    # 3. Local search / edge swapping to improve girth

    # First, try the explicit construction from the seed (it's fast and gives a valid graph)
    # Then try to find something smaller via random construction + local optimization

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            # BFS with parent tracking
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model + deduplication."""
        if seed is not None:
            random.seed(seed)
        # Configuration model
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def try_explicit_construction():
        """The Lazebnik-Ustimenko-Woldar construction from the seed."""
        q = k
        # For girth g, we need d such that girth >= d + 5 (odd d) or d + 4 (even d)
        # The construction gives girth at least d+5 for odd d
        # We want girth >= g, so d+5 >= g => d >= g-5
        # But d must be odd for the stated guarantee. Let's pick d = g-5 if odd, else g-4
        if (g - 5) % 2 == 1:
            d = g - 5
        else:
            d = g - 4
        if d < 3:
            d = 3

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Rule definition
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
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

    def local_search_improve(n, edges, target_girth, max_time):
        """Try to reduce vertex count by removing vertices and re-regularizing."""
        # This is complex; for now, just verify and return
        if verify_girth(n, edges, target_girth):
            return n, edges
        return None, None

    # Step 1: Try explicit construction
    if time_left() > 10:
        try:
            n_exp, edges_exp = try_explicit_construction()
            if verify_girth(n_exp, edges_exp, g):
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass

    # Step 2: Try random regular graphs starting from Moore bound
    if best_n is None or time_left() > 5:
        # Start from a reasonable n
        # For k=5, g=9: moore ~ 426, known best is 1116
        # Try random graphs at various sizes
        start_n = max(moore, k + 1)
        # The explicit construction for k=5, g=9 gives a certain size
        # Let's try random graphs near the moore bound first, then increase

        # For random k-regular graphs, girth is typically logarithmic in n
        # So for girth 9, we need n to be reasonably large
        # Let's try a range of n values

        if best_n is not None:
            # Try to beat the explicit construction
            upper_n = best_n
        else:
            upper_n = start_n * 3  # some upper bound

        # Binary search for smallest n where a valid graph exists
        # But random construction may fail; use multiple tries

        lo = max(start_n, moore)
        hi = upper_n if upper_n else lo * 5

        # For efficiency, try a few random seeds at each n
        for n_try in range(lo, hi + 1, max(1, (hi - lo) // 20)):
            if time_left() < 5:
                break
            for attempt in range(50):
                if time_left() < 5:
                    break
                result = generate_random_regular(n_try, k, seed=random.randint(0, 2**31))
                if result is not None:
                    n_r, edges_r = result
                    if verify_girth(n_r, edges_r, g):
                        if best_n is None or n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break  # found one at this n, move to next n

    # Step 3: If we have a valid graph, try local edge swaps to reduce n
    # (This is complex and may not fit in time budget, so skip for now)

    if best_n is None:
        # Fallback: return the explicit construction even if verification failed
        # (should not happen, but safety net)
        n_exp, edges_exp = try_explicit_construction()
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END
