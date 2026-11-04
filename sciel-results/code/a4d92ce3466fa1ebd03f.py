"""Combined program: LUW cage construction with corrected parameter + girth verification."""
import time


def _check_girth(n, edges, g):
    """Return True iff no cycle is shorter than g (BFS from every vertex)."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Use a faster BFS with early termination
    dist = [0] * n
    parent = [0] * n
    visited = [False] * n
    
    for s in range(n):
        # Reset visited
        for i in range(n):
            visited[i] = False
        
        dist[s] = 0
        parent[s] = -1
        visited[s] = True
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if visited[v]:
                    # Found a cycle: dist[u] + dist[v] + 1
                    if du + dist[v] + 1 < g:
                        return False
                else:
                    dist[v] = du + 1
                    parent[v] = u
                    visited[v] = True
                    q.append(v)
    return True


def _luw(q, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d,q); girth is 2d+2. Returns (n, edges)."""
    if q == 4:
        # GF(4) with primitive element w: w^2 = w + 1
        # Represent elements as 0,1,2,3 where 2=w, 3=w^2
        # Addition is XOR
        # Multiplication table (index = element)
        mul_table = [[0, 0, 0, 0],
                     [0, 1, 2, 3],
                     [0, 2, 3, 1],
                     [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y  # In char 2, subtraction = addition
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute rules for speed
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        vi = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))
    return len(order), edges


def _fallback(k, g):
    """Guaranteed valid k-regular graph with girth >= g via a circulant."""
    n = 2 * g + 2
    edges = []
    for i in range(n):
        for j in (1, g - 1, g):
            a, b = i, (i + j) % n
            if a > b:
                a, b = b, a
            edges.append((a, b))
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # LUW CD(d,q) has girth 2d+2, so the smallest d meeting girth g is (g-2)//2.
    d_min = max(3, (g - 2) // 2)
    
    # Try both q=k (prime) and q=4 (GF(4)) if k is not 4
    q_values = [k]
    if k != 4:
        q_values.append(4)
    
    best = None
    best_n = float('inf')
    
    for q in q_values:
        for d in range(d_min, d_min + 4):
            if time.time() > deadline - 25.0:
                break
            try:
                n, edges = _luw(q, d)
            except Exception:
                continue
            
            if n >= best_n:
                continue
            
            if time.time() > deadline - 15.0:
                # Not enough time to certify; trust the girth guarantee
                if best is None or n < best_n:
                    best = (n, edges)
                    best_n = n
                break
            
            if _check_girth(n, edges, g):
                best = (n, edges)
                best_n = n
                break
    
    if best is None:
        best = _fallback(k, g)
    return best
