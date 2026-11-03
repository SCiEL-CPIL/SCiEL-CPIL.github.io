import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    def has_short_cycle(n, adj, g):
        """Check if graph has any cycle of length < g. Returns True if it does."""
        half = (g - 1) // 2
        for start in range(n):
            # BFS from start, tracking parent to avoid going back
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            q = deque([start])
            visited[start] = start
            while q:
                u = q.popleft()
                if depth[u] >= half:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and visited[v] != u:
                        # Found a cycle. Compute its length.
                        # Trace back from u and v to find LCA
                        path_u = set()
                        curr = u
                        while curr != -1:
                            path_u.add(curr)
                            curr = parent[curr]
                        curr = v
                        lca = -1
                        while curr != -1:
                            if curr in path_u:
                                lca = curr
                                break
                            curr = parent[curr]
                        if lca == -1:
                            # Should not happen in connected component
                            continue
                        cycle_len = depth[u] + depth[v] - 2 * depth[lca] + 1
                        if cycle_len < g:
                            return True
        return False

    def generate_random_3_regular(n):
        """Generate a random 3-regular graph using configuration model."""
        if n % 2 != 0:
            n += 1
        stubs = [i for i in range(n) for _ in range(3)]
        max_attempts = 1000
        for _ in range(max_attempts):
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, 3 * n, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return n, adj
        return None

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Moore bound for (3, g)
    moore = 1 + 3 * (2 ** ((g - 1) // 2) - 1)
    # Start from a reasonable size and increase
    n_start = max(moore, 766)
    if n_start % 2 != 0:
        n_start += 1
    
    # Try increasing sizes
    for n in range(n_start, n_start + 200, 2):
        if time.time() - start_time > time_limit:
            break
        # Try multiple random graphs
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            result = generate_random_3_regular(n)
            if result is None:
                continue
            n_actual, adj = result
            if not has_short_cycle(n_actual, adj, g):
                return n_actual, edges_from_adj(n_actual, adj)
    
    # Fallback: return a known valid construction if time runs out
    # Use the Cayley graph approach as fallback
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
