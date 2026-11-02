import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    if k != 3:
        return seed_fallback(k, g)

    target_girth = g  # 16

    def has_short_cycle(adj, n, max_cycle_len):
        """Check if there's a cycle shorter than max_cycle_len. Returns (True, cycle) or (False, None)."""
        max_depth = max_cycle_len // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_cycle_len:
                            # Reconstruct cycle
                            path_u = []
                            cur = u
                            while cur != s:
                                path_u.append(cur)
                                cur = parent[cur]
                                if cur == -1:
                                    break
                            path_u.reverse()
                            path_v = []
                            cur = v
                            while cur != s:
                                path_v.append(cur)
                                cur = parent[cur]
                                if cur == -1:
                                    break
                            path_v.reverse()
                            cycle = path_u + path_v
                            return True, cycle
        return False, None

    def generate_random_3_regular(n, rng):
        """Generate a random simple 3-regular graph using configuration model with rejection."""
        for _ in range(200):
            stubs = []
            for i in range(n):
                for _ in range(3):
                    stubs.append(i)
            rng.shuffle(stubs)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, 3*n, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            ok = True
            for i in range(n):
                if len(adj[i]) != 3:
                    ok = False
                    break
            if ok:
                return adj
        return None

    # Main logic
    rng = random.Random(42)
    
    # Try to construct a 3-regular graph with girth >= target_girth
    # For n=936, random graphs won't have girth 16, so we fall back to LUW
    # But let's try a few different n values with random generation + check
    
    for n_try in [936, 900, 800, 700, 600, 512]:
        if time.time() > deadline - 20:
            break
        adj = generate_random_3_regular(n_try, rng)
        if adj is None:
            continue
        has_sc, _ = has_short_cycle(adj, n_try, target_girth)
        if not has_sc:
            edges = []
            for i in range(n_try):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n_try, edges
    
    # Fallback: LUW
    return seed_fallback(k, g)

def seed_fallback(k, g):
    """LUW construction fallback."""
    if k != 3:
        n = 2 * k
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    q = 3
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    idx = 0
    while idx < len(order):
        vertex = order[idx]
        idx += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
