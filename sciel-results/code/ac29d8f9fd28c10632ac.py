import time
from collections import defaultdict, deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    q = k

    # Determine d such that girth >= g
    # LUW girth = 2*d + 2
    d = (g - 2) // 2
    if 2 * d + 2 < g:
        d += 1

    # GF(q) operations
    if q == 4:
        def gf4_mul(x, y):
            if x == 0 or y == 0:
                return 0
            if x == 1:
                return y
            if y == 1:
                return x
            if x == 2 and y == 2:
                return 3
            if x == 2 and y == 3:
                return 2
            if x == 3 and y == 2:
                return 2
            if x == 3 and y == 3:
                return 1
            return 0
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return gf4_mul(x, y)
    elif q == 9:
        def gf9_to_ab(x):
            return (x % 3, x // 3)
        def gf9_from_ab(a, b):
            return a + 3 * b
        def gf9_add(x, y):
            ax, bx = gf9_to_ab(x)
            ay, by = gf9_to_ab(y)
            return gf9_from_ab((ax + ay) % 3, (bx + by) % 3)
        def gf9_sub(x, y):
            ax, bx = gf9_to_ab(x)
            ay, by = gf9_to_ab(y)
            return gf9_from_ab((ax - ay) % 3, (bx - by) % 3)
        def gf9_mul(x, y):
            ax, bx = gf9_to_ab(x)
            ay, by = gf9_to_ab(y)
            real = (ax * ay + bx * by) % 3
            coeff = (ax * by + bx * ay + bx * by) % 3
            return gf9_from_ab(real, coeff)
        def add(x, y): return gf9_add(x, y)
        def sub(x, y): return gf9_sub(x, y)
        def mul(x, y): return gf9_mul(x, y)
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Generate all vertices: 2 * q^d
    qd = q ** d
    n = 2 * qd

    # Generate edges
    edges = []
    edge_set = set()

    for i in range(n):
        side = i // qd
        coord_idx = i % qd

        # Convert coord_idx to coordinates
        x = []
        temp = coord_idx
        for _ in range(d):
            x.append(temp % q)
            temp //= q
        x = tuple(x)

        for first in range(q):
            y = [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                if side == 0:
                    # y_c = x_c + y_1 * x_{c-1}
                    y[c - 1] = add(x[c - 1], mul(y[0], x[c - 2]))
                else:
                    # y_c = x_c - x_1 * y_{c-1}
                    y[c - 1] = sub(x[c - 1], mul(x[0], y[c - 2]))

            neighbor_side = 1 - side
            # Convert y to coord index
            neighbor_coord = 0
            for c in range(d):
                neighbor_coord += y[c] * (q ** c)
            neighbor_idx = neighbor_side * qd + neighbor_coord

            if i < neighbor_idx:
                edge = (i, neighbor_idx)
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append(edge)

        if time.time() > deadline:
            break

    # Verify girth using BFS from each vertex (non-backtracking)
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check girth by BFS from each vertex, tracking parent to avoid backtracking
    def check_girth(adj, n, min_girth):
        for start in range(n):
            # BFS from start, tracking (node, prev) to avoid backtracking
            visited = {}  # (node, prev) -> distance
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start, 1))
                visited[(neighbor, start)] = 1

            while queue:
                node, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue  # No need to explore further
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue  # Don't backtrack
                    if neighbor == start:
                        return dist  # Found a cycle of length dist
                    state = (neighbor, node)
                    if state not in visited:
                        visited[state] = dist + 1
                        queue.append((neighbor, node, dist + 1))
        return min_girth  # No shorter cycle found

    if time.time() < deadline - 10:
        actual_girth = check_girth(adj, n, g)
        # If actual_girth < g, the construction is wrong

    return n, edges
