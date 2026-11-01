import time
import random
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for start in range(n):
        # dist[state] = shortest non-backtracking walk length from start to state
        # state = (current_vertex, previous_vertex)
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(base_edges, m, shifts):
    """Build a voltage graph lift over Z_m."""
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Heawood graph (14 vertices, 21 edges, girth 6)
    heawood_edges = [
        (0, 7), (0, 8), (0, 9),
        (1, 7), (1, 10), (1, 11),
        (2, 7), (2, 12), (2, 13),
        (3, 8), (3, 10), (3, 13),
        (4, 8), (4, 11), (4, 12),
        (5, 9), (5, 10), (5, 12),
        (6, 9), (6, 11), (6, 13)
    ]

    # Dodecahedral graph (20 vertices, 30 edges, girth 5)
    dodeca_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15)
    ]

    # Petersen graph (10 vertices, 15 edges, girth 5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    base_graphs = [
        ("heawood", heawood_edges, 14),
        ("dodeca", dodeca_edges, 20),
        ("petersen", petersen_edges, 10)
    ]

    best_result = None
    best_n = float('inf')

    for name, base_edges, n_base in base_graphs:
        if time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)
        
        # Try m values
        for m in range(n_min, n_min + 100):
            if time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 3:
                    break
                rng = random.Random(hash((name, m, attempt)))
                shifts_rand = [rng.randint(0, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

    # Fallback: LUW
    if time.time() < deadline - 5:
        for d_try in [7, 8, 9, 10]:
            if time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
