import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback: LUW construction from seed
    def luw_fallback():
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

    # Heuristic Construction
    def try_construct(N, k, g, timeout):
        if N * k % 2 != 0:
            return None
        adj = [[] for _ in range(N)]
        deg = [0] * N
        edges = []
        
        # Helper to check if adding (u,v) creates a short cycle
        def creates_short_cycle(u, v):
            # BFS from u to v
            # We want to know if there is a path of length < g-1
            # If dist(u,v) < g-1, then adding (u,v) creates cycle of length dist+1 < g
            # Note: u and v are not yet connected.
            queue = collections.deque()
            visited = {u: 0}
            queue.append(u)
            while queue:
                curr = queue.popleft()
                dist = visited[curr]
                if dist >= g - 1:
                    continue
                for neighbor in adj[curr]:
                    if neighbor == v:
                        # Found path of length dist+1
                        if dist + 1 < g:
                            return True
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append(neighbor)
            return False

        # List of vertices that still need edges
        # We'll just pick random pairs until we have enough edges
        total_edges_needed = N * k // 2
        attempts = 0
        max_attempts = 100000
        
        while len(edges) < total_edges_needed:
            if time.time() > timeout:
                return None
            if attempts > max_attempts:
                return None
            attempts += 1
            
            # Pick two random vertices with deg < k
            # Optimization: maintain a list of available vertices?
            # For N=1000, picking random is O(1) but might pick full vertices.
            # Let's just pick random u, v and check.
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            
            if u == v:
                continue
            if deg[u] >= k or deg[v] >= k:
                continue
            if v in adj[u]: # Already connected
                continue
                
            if creates_short_cycle(u, v):
                continue
                
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            
        # Verify all degrees are k
        if any(d != k for d in deg):
            return None
            
        # Optional: Verify girth is indeed >= g
        # The construction ensures no cycle < g is created.
        # But we should be safe.
        
        return edges

    # Try increasing N
    # Start with known best if applicable, or Moore bound
    # For k=5, g=9, known best is 1116.
    # We will try a range of N values.
    
    # Determine candidate N values
    candidates = []
    # Moore bound approximation
    # n >= 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # For g=9, g/2-1 = 3.
    # n >= 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # We know 1116 works.
    # Let's try values around 1116 and upwards.
    
    if k == 5 and g == 9:
        candidates = [1116, 1150, 1200, 1300, 1500, 2000, 3000, 5000]
    else:
        # Generic heuristic: start from Moore bound * 2, go up
        moore = 1
        term = 1
        for i in range(g // 2):
            moore += k * term
            term *= (k - 1)
        # Add some buffer
        candidates = []
        n = moore * 2
        while n < 10000:
            candidates.append(n)
            n += max(10, n // 10)
            
    for N in candidates:
        if time.time() > deadline:
            break
        # Ensure N*k is even
        if (N * k) % 2 != 0:
            N += 1
        edges = try_construct(N, k, g, deadline)
        if edges is not None:
            return N, edges

    # Fallback to LUW
    return luw_fallback()
