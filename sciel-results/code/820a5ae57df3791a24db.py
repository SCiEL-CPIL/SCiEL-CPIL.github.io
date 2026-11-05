"""Cage graph constructor: 3-regular, girth >= g, minimize n."""
import time
import sys

_START = time.time()
_DEADLINE = _START + 140  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _verify_girth(n, adj, g):
    """Verify girth >= g using BFS with parent tracking. Returns True if girth >= g."""
    half = g // 2
    for start in range(n):
        if _deadline_hit():
            return False
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= half:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a non-tree edge creating a cycle
                    # Cycle length is du + dist[v] + 1
                    # But only if this isn't just the parent edge (already skipped)
                    # In a tree, edges connect level d to d+1 or d-1
                    # If dist[v] < du, v is an ancestor (not parent) -> cycle
                    # If dist[v] == du, cross edge -> cycle (but in regular graph from BFS tree)
                    # If dist[v] > du, shouldn't happen in BFS (v would be discovered later)
                    # Actually in BFS, for neighbor v of u at depth du:
                    #   dist[v] is either du-1 (parent), du (cross), du+1 (child/unvisited->now visited)
                    #   or -1 (unvisited)
                    # Since we skip parent, if dist[v] != -1 and v != parent[u]:
                    #   cycle_len = du + dist[v] + 1
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _find_prime_near(target):
    """Find a prime near target (>= target)."""
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    p = target
    while not is_prime(p):
        p += 1
    return p


def _margulis_graph(q, r, s):
    """
    Construct the Margulis (LUB) graph on Z_q x Z_q.
    Each vertex (a, b) connects to:
      (a+r, b+s), (a-r, b-s), (a+s, b+r), (a-s, b-r)
    This gives a 4-regular graph. For k=3, we need a different approach.
    Actually for 3-regular, we can use a different construction.
    """
    n = q * q
    adj = [[] for _ in range(n)]
    for a in range(q):
        for b in range(q):
            u = a * q + b
            # 4 neighbors for LUB
            nbrs = [
                ((a + r) % q, (b + s) % q),
                ((a - r) % q, (b - s) % q),
                ((a + s) % q, (b + r) % q),
                ((a - s) % q, (b - r) % q),
            ]
            for na, nb in nbrs:
                v = na * q + nb
                if v != u and v not in adj[u]:
                    adj[u].append(v)
                    adj[v].append(u)
    return n, adj


def _construct_3reg_voltage_lift(m, voltages):
    """
    Construct 3-regular graph as voltage lift of Petersen graph.
    Petersen: 10 vertices, 15 edges, 3-regular.
    Lift by Z_m: 10*m vertices.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n = 10 * m
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(petersen_edges):
        vol = voltages[i]
        for l in range(m):
            u_l = u * m + l
            v_l = v * m + (l + vol) % m
            adj[u_l].append(v_l)
            adj[v_l].append(u_l)
    return n, adj


def _check_girth_fast(n, adj, g):
    """Fast girth check using BFS with parent tracking."""
    half = g // 2
    for start in range(n):
        if _deadline_hit():
            return False
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= half:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def construct(k, g):
    """Construct k-regular graph with girth >= g on as few vertices as possible."""
    if k == 3 and g == 16:
        # Strategy: Voltage lift of Petersen graph with optimized search
        # Petersen graph has girth 5. Lift by Z_m with appropriate voltages
        # can achieve much higher girth.
        
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Try m values starting from minimum that could work
        # Moore bound for (3,16): 2 + 3*(2^7 - 1) = 2 + 3*127 = 383, but actual lower bound is 512
        # n = 10*m, so m >= 52 for n >= 520
        # But we can also try smaller m with better voltage assignments
        
        import random
        random.seed(12345)
        
        best_n = None
        best_edges = None
        
        # Try deterministic structured voltage assignments first
        # Then random search
        
        for m in range(51, 70):
            if _deadline_hit():
                break
            
            # Try structured voltages: all edges get voltage 1 or based on position
            # Structure 1: alternating voltages
            for pattern in range(4):
                if _deadline_hit():
                    break
                voltages = []
                for i in range(15):
                    if pattern == 0:
                        voltages.append(1)
                    elif pattern == 1:
                        voltages.append((i % m) + 1)
                    elif pattern == 2:
                        voltages.append((i * 3 + 1) % m)
                    else:
                        voltages.append((i * 7 + 2) % m)
                
                n, adj = _construct_3reg_voltage_lift(m, voltages)
                if _check_girth_fast(n, adj, 16):
                    edges = []
                    for i, (u, v) in enumerate(petersen_edges):
                        vol = voltages[i]
                        for l in range(m):
                            u_l = u * m + l
                            v_l = v * m + (l + vol) % m
                            edges.append((u_l, v_l))
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    return n, edges
            
            # Random voltage search - limited attempts per m
            n_attempts = 20 if m <= 55 else 10
            for attempt in range(n_attempts):
                if _deadline_hit():
                    break
                voltages = [random.randint(1, m - 1) for _ in range(15)]
                n, adj = _construct_3reg_voltage_lift(m, voltages)
                if _check_girth_fast(n, adj, 16):
                    edges = []
                    for i, (u, v) in enumerate(petersen_edges):
                        vol = voltages[i]
                        for l in range(m):
                            u_l = u * m + l
                            v_l = v * m + (l + vol) % m
                            edges.append((u_l, v_l))
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    return n, edges
        
        # If no voltage lift worked, use LUW construction as fallback
        q = 3
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

    # Generic fallback for other k, g
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
