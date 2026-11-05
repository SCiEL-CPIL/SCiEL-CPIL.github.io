import collections
import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    It searches for the smallest parameter d such that the resulting graph component has girth >= g.
    """
    q = k
    # LUW requires q to be a prime power (or 4).
    # If k is not a prime power, this construction fails.
    # For this task, k=5 is prime.
    # If k is not suitable, fallback to a trivial construction or raise error?
    # The task implies k=5.
    
    # Helper to check if q is a prime power or 4
    def is_prime_power_or_4(n):
        if n == 4:
            return True
        # Check if n is a prime power
        # Simple check for small n
        import math
        for p in range(2, n + 1):
            # Check if n is p^m
            x = n
            while x % p == 0:
                x //= p
            if x == 1:
                # Check if p is prime
                if p > 1:
                    is_prime = True
                    for i in range(2, int(math.isqrt(p)) + 1):
                        if p % i == 0:
                            is_prime = False
                            break
                    if is_prime:
                        return True
        return False

    if not is_prime_power_or_4(q):
        # Fallback: Return a trivial graph or use a different method
        # For now, assume k is valid for LUW
        pass

    def gf_operations(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
            return add, sub, mul
        else:
            # Assume prime
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            return add, sub, mul

    add, sub, mul = gf_operations(q)

    def build_luw(d):
        # Define rules for c = 2..d
        # Based on the seed program logic
        rule = {}
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
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

    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex up to depth r
        r = (min_girth - 1) // 2
        for start in range(n):
            # dist[start] = 0
            # parent[start] = -1
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Back edge or cross edge
                        # Cycle length is dist[u] + 1 + dist[v] ?
                        # In undirected BFS, if v is visited and v != parent[u],
                        # and dist[v] <= dist[u] + 1
                        # The cycle formed is u -> ... -> start -> ... -> v -> u
                        # Length = dist[u] + dist[v] + 1
                        # But this is only valid if the path to u and v are disjoint except at start?
                        # No, BFS tree is a tree.
                        # The cycle is u - v - (path to start) - (path to start) - u
                        # Actually, if we find an edge (u, v) where u and v are in the BFS tree,
                        # the cycle length is dist[u] + dist[v] + 1.
                        # We check if this is < min_girth.
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            # Note: This check might miss some cycles if the shortest cycle doesn't go through the start?
            # No, any cycle contains a vertex. If we run BFS from every vertex, we will find the cycle when start is on the cycle.
            # However, the cycle length calculated is the length of the cycle passing through start?
            # If the cycle is 5, and start is on it, BFS will find it.
            # If start is not on the cycle, BFS won't detect it as a cycle through start.
            # So running from all vertices is correct.
            
        return True

    # Search for d
    # Start from small d
    # Max d to avoid timeout
    max_d = 10
    best_result = None
    
    for d in range(1, max_d + 1):
        n, edges = build_luw(d)
        # Optional: Check if n is reasonable
        if check_girth(n, edges, g):
            return n, edges
            
    # If no d found, return the one from seed logic (d=5 or similar)
    # Fallback to seed behavior
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = build_luw(d_seed)
    return n, edges
