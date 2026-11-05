import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

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

    n_min = moore_bound(k, g)

    def check_girth_bfs(adj, n, g):
        """Check girth using BFS from each vertex - no cycle shorter than g."""
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        if dist[x] + dist[y] + 1 < g:
                            return False
        return True

    def is_connected(adj, n):
        if n == 0:
            return True
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return None
        if not is_connected(adj, n):
            return None
        if not check_girth_bfs(adj, n, g):
            return None
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Approach 1: Cayley graph on A_n (alternating group) with involutions ----------
    # For k=5, g=9, we need a group with 5 involutory generators
    # such that the girth is at least 9.
    # Use A_7 or A_8 which have many involutions.

    def cayley_group_search(group_size, k, g, time_budget):
        """
        Try to find k involutory generators of a group of given size
        such that the Cayley graph has girth >= g.
        We'll use a representation of the group as permutations.
        """
        # For efficiency, we'll work with permutation groups
        # Represent group elements as tuples (permutations)
        # This is only feasible for small groups

        # For A_7: 2520 elements, involutions are products of disjoint transpositions
        # Actually, in A_7, involutions are products of 2 or 3 disjoint transpositions
        # (must be even permutation)

        # Let's use a different approach: work with a specific group structure
        # Use the group as a list of permutations, and search for good generators

        # For practical purposes with time limits, let's use a randomized approach
        # on a known group structure

        # Approach: Use A_8 (20160 elements) or similar
        # But generating the full group is expensive.

        # Alternative: Use the fact that for k=5, g=9, n=1116 is known.
        # 1116 = 4 * 279 = 4 * 3 * 93 = 4 * 3 * 3 * 31
        # Or 1116 = 2^2 * 3^2 * 31

        # Let's try a different approach: use a voltage graph / covering construction

        return None

    # ---------- Approach 2: Use known constructions for small (k,g) ----------
    # For k=5, g=9: The best known is 1116 vertices (Exoo et al. 2026)
    # The Moore bound is 428.

    # Let's try to construct using a product of smaller graphs or a known method.

    # Approach: Use the "lifting" technique or voltage graphs

    # ---------- Approach 3: Random regular graph with girth check ----------
    # Generate random k-regular graphs and check girth

    def random_k_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices."""
        if seed is not None:
            random.seed(seed)

        # Use the configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)

        random.shuffle(half_edges)

        # Pair up half-edges to form edges
        adj = [set() for _ in range(n)]
        edges = []
        used = [False] * len(half_edges)

        i = 0
        while i < len(half_edges):
            if used[i]:
                i += 1
                continue
            j = i + 1
            while j < len(half_edges) and (used[j] or half_edges[j] == half_edges[i]):
                j += 1
            if j >= len(half_edges):
                # Retry
                return None
            u, v = half_edges[i], half_edges[j]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                used[i] = True
                used[j] = True
            i += 1

        # Check all vertices have degree k
        for i in range(n):
            if len(adj[i]) != k:
                return None

        return adj, edges

    def try_random_regular(n, k, g, attempts, time_budget):
        """Try random k-regular graphs of size n until one has girth >= g."""
        start = time.time()
        for attempt in range(attempts):
            if time.time() - start > time_budget:
                break
            result = random_k_regular(n, k)
            if result is None:
                continue
            adj, edges = result
            if check_girth_bfs(adj, n, g):
                return (n, edges)
        return None

    # ---------- Approach 4: Use the Lazebnik-Ustimenko-Woldar construction ----------
    def luw_construct(k, g):
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

    # ---------- Main search strategy ----------
    # Start with the LUW construction as a baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # Try to find smaller graphs using random regular graphs
    # The Moore bound is n_min, so we search from n_min up to best_n - 1

    # For k=5, g=9: n_min = 428, best known = 1116
    # We'll try random regular graphs in a range

    # First, try a few sizes around the Moore bound
    # But random regular graphs of small size are unlikely to have large girth
    # So we focus on sizes that are plausible

    # For girth 9, we need n to be sufficiently large
    # The probability that a random 5-regular graph on n vertices has girth >= 9
    # decreases as n gets smaller.

    # Let's try sizes from max(n_min, 500) to best_n - 1
    # But we need to be careful about time

    search_start = max(n_min, 450)
    search_end = min(best_n - 1, 800)  # Don't search too far

    # Try a few sizes with limited attempts
    time_per_size = 10.0  # seconds per size

    for n_try in range(search_start, search_end + 1, 10):  # Step by 10 for speed
        if time.time() > deadline - 10:
            break
        if n_try % 2 != 0 and k * n_try % 2 != 0:
            continue  # k*n must be even for k-regular graph
        result = try_random_regular(n_try, k, g, attempts=20, time_budget=time_per_size)
        if result is not None:
            best_n = n_try
            best_result = result
            break  # Found a valid graph, stop searching smaller

    # If we haven't found anything smaller, return the LUW result
    return best_result

# EVOLVE-BLOCK-END
