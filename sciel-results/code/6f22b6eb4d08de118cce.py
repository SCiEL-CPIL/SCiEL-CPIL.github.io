"""Combined program: algebraic voltage lifts + fast verification + random 3-regular fallback + LUW."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
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
    """Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid."""
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

def _random_3reg_graph(n, rng):
    """Generate a random 3-regular graph on n vertices. Returns (adj, edges) or None."""
    if n < 4 or n % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    if not valid:
        return None
    # Check connectedness
    visited = set([0])
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
    return adj, edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Base graphs
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    cube_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    base_graphs = [
        ("petersen", petersen_edges, 10),
        ("k33", k33_edges, 6),
        ("cube", cube_edges, 8),
    ]

    best_result = None
    best_n = float('inf')

    # Phase 1: Voltage lifts with deterministic + random shifts
    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 20:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        for m in range(n_min, n_min + 150):
            if _time.time() > deadline - 15:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # Deterministic shifts: quadratic residues
            residues = set()
            for a in range(1, m):
                residues.add((a * a) % m)
                residues.add((m - (a * a) % m) % m)
            residues_list = sorted(residues)
            if len(residues_list) < e_count:
                residues_list = list(range(m))
            step = max(1, len(residues_list) // e_count)
            shifts_det = [residues_list[(i * step) % len(residues_list)] for i in range(e_count)]
            result = _build_lift(base_edges, m, shifts_det)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    return n_r, edges

            # Golden ratio shifts
            phi = 0.6180339887498948
            shifts_gold = [int(((i + 1) * phi * m) % m) for i in range(e_count)]
            result = _build_lift(base_edges, m, shifts_gold)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    return n_r, edges

            # Random shifts with diverse seeds
            max_attempts = 40
            for attempt in range(max_attempts):
                if _time.time() > deadline - 10:
                    break
                seed_val = hash((name, m, attempt))
                rng = _random.Random(seed_val)
                shifts_rand = [rng.randint(1, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        return n_r, edges

    # Phase 2: Random 3-regular graphs (from inspiration)
    if _time.time() < deadline - 10:
        n_try = max(moore, 100)
        if n_try % 2 != 0:
            n_try += 1
        while n_try < 2000 and _time.time() < deadline - 5:
            for attempt in range(30):
                if _time.time() > deadline - 3:
                    break
                rng = _random.Random(hash(("rand3reg", n_try, attempt)))
                res = _random_3reg_graph(n_try, rng)
                if res is None:
                    continue
                adj, edges = res
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n_try, g):
                    return n_try, edges
            n_try += 16

    # Phase 3: LUW fallback
    if _time.time() < deadline - 5:
        for d_try in range(6, 12):
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth_fast(adj_list, n_try, g):
                        return n_try, edges_try
            except Exception:
                pass

    # Final fallback
    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
