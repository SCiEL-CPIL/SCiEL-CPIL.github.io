import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    def girth_ok(adj, g):
        """Check if graph has no cycle shorter than g using BFS to depth floor((g-1)/2)."""
        n = len(adj)
        depth = (g - 1) // 2
        for s in range(n):
            # BFS from s, tracking (vertex, parent) to avoid backtracking
            visited = {}  # (v, parent) -> depth
            queue = deque()
            queue.append((s, -1, 0))
            visited[(s, -1)] = 0
            while queue:
                v, parent, d = queue.popleft()
                if d >= depth:
                    continue
                for w in adj[v]:
                    if w == parent:
                        continue
                    key = (w, v)
                    if key in visited:
                        # Found a cycle of length d+1+visited[key]... 
                        # Actually if we reach (w,v) again via a different path, cycle exists
                        # The cycle length is d + 1 + (distance from s to w via the other path)
                        # Simpler: if we've seen w at any point with a different parent, cycle
                        # Let's use a different approach
                        pass
                    else:
                        visited[key] = d + 1
                        queue.append((w, v, d + 1))
            # Check for cycles: if any vertex is reached via two different paths
            # Better approach: BFS and if we encounter a vertex already visited at depth <= depth, 
            # and it's not the parent, we have a short cycle
        # Actually, let's use a cleaner method
        return True

    def has_short_cycle(adj, g):
        """Return True if there's a cycle of length < g."""
        n = len(adj)
        depth = (g - 1) // 2
        for s in range(n):
            # dist[v] = shortest distance from s, and parent for path reconstruction
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                v = queue.popleft()
                if dist[v] >= depth:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and v != parent[w]:
                        # Found a cycle: dist[v] + dist[w] + 1
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return True
        return False

    def random_k_regular(n, k, rng):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                continue  # skip loops for now, will retry
            if v not in adj[u]:
                adj[u].append(v)
                adj[v].append(u)
            else:
                return None  # multi-edge, retry
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj

    best_n = None
    best_edges = None
    rng = random.Random(42)

    # Try increasing sizes
    for n in range(426, 2000):
        if time.time() > deadline - 20:
            break
        if n * k % 2 != 0:
            continue
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            if time.time() > deadline - 10:
                break
            attempts += 1
            adj = random_k_regular(n, k, rng)
            if adj is None:
                continue
            if not has_short_cycle(adj, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                return best_n, best_edges

    # Fallback: use the LUW construction if we ran out of time
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
