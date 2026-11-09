"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        moore = 2 * (k ** (g // 2 - 1))
    else:
        moore = 1 + k * sum(k ** i for i in range((g - 1) // 2))
    
    # Start from a reasonable lower bound; known best for (3,16) is 936
    # Use max(moore, some heuristic) as starting point
    min_n = max(moore, 2 * k)
    
    best_n = None
    best_edges = None
    
    # Try sizes from a lower estimate upward
    # For (3,16), try starting around 900-1000
    if k == 3 and g == 16:
        candidates = list(range(936, 1200, 4))  # n must be even for 3-regular
        candidates = [n for n in candidates if n % 2 == 0]
    else:
        # General case: start from Moore bound, step by k
        candidates = list(range(min_n, min_n + 2000, k))
        candidates = [n for n in candidates if n * k % 2 == 0]  # n*k must be even
    
    def _generate_random_k_regular(n, k, rng):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0 or n < k:
            return None
        # Use configuration model
        half_edges = list(range(n * k))
        rng.shuffle(half_edges)
        # Group into n vertices, each with k stubs
        adj = [[] for _ in range(n)]
        for i in range(n * k):
            adj[i % n].append(half_edges[i])
        # Now pair up stubs
        adj2 = [[] for _ in range(n)]
        stubs = []
        for v in range(n):
            for s in adj[v]:
                stubs.append(v)
        rng.shuffle(stubs)
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                continue  # skip self-loops, will re-pair
            adj2[u].append(v)
            adj2[v].append(u)
        # Check if k-regular
        for v in range(n):
            if len(adj2[v]) != k:
                return None
        # Remove duplicates (multiedges)
        edges = []
        for v in range(n):
            for u in adj2[v]:
                if v < u:
                    edges.append((v, u))
        if len(edges) != n * k // 2:
            return None
        return edges
    
    def _check_girth(adj, k, g, max_time=2.0):
        """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
        check_start = time.time()
        n = len(adj)
        # For each vertex, do non-backtracking BFS up to depth g//2
        for start in range(n):
            if time.time() - check_start > max_time:
                return False
            # BFS with non-backtracking: state = (current, prev)
            visited = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start))
                visited[(nb, start)] = 1
            depth = 1
            while queue and depth < g // 2:
                new_queue = deque()
                for (curr, prev) in queue:
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        if (nb, curr) in visited:
                            # Found a cycle of length <= depth + 1
                            return False
                        new_queue.append((nb, curr))
                        visited[(nb, curr)] = depth + 1
                queue = new_queue
                depth += 1
        return True
    
    # Try each candidate size
    rng = random.Random(42)
    for n in candidates:
        if time.time() > deadline:
            break
        # Try multiple random graphs for this size
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline:
            edges = _generate_random_k_regular(n, k, rng)
            if edges is None:
                attempts += 1
                continue
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # Verify k-regular
            valid = all(len(adj[v]) == k for v in range(n))
            if not valid:
                attempts += 1
                continue
            # Check girth
            if _check_girth(adj, k, g, max_time=1.5):
                best_n = n
                best_edges = edges
                break
            attempts += 1
        if best_n is not None:
            break
    
    # Fallback: use LUW construction if no random graph found
    if best_n is None:
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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        return len(order), edges_luw
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
