"""Combined program: algebraic voltage lifts + fast verification + LUW fallback + random regular graph search."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _quadratic_residues(m, count):
    """Generate count distinct quadratic residues modulo m, spread out."""
    if m <= 1:
        return [0] * count
    residues = set()
    for a in range(1, m):
        residues.add((a * a) % m)
    for a in range(1, m):
        residues.add((m - (a * a) % m) % m)
    residues_list = sorted(residues)
    if len(residues_list) < count:
        residues_list = list(range(m))
    step = max(1, len(residues_list) // count)
    result = []
    for i in range(count):
        idx = (i * step) % len(residues_list)
        result.append(residues_list[idx])
    return result

def _golden_ratio_shifts(m, count):
    """Generate shifts using golden ratio conjugate for good spread."""
    if m <= 1:
        return [0] * count
    phi = 0.6180339887498948
    result = []
    for i in range(count):
        val = int(((i + 1) * phi * m) % m)
        result.append(val)
    return result

def _modular_pattern_shifts(m, count, base_mod):
    """Generate shifts using modular pattern with different base modulus."""
    if m <= 1:
        return [0] * count
    result = []
    for i in range(count):
        val = ((i + 1) * base_mod) % m
        result.append(val)
    return result

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
    n_base = max(max(u for u, v in base_edges), max(v for u, v in base_edges)) + 1
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

def _generate_cubic_random(n, seed=None):
    """Generate a random 3-regular graph on n vertices using configuration model.
    Returns (adj_list, edge_list) or None if not simple."""
    if n < 4 or n % 2 != 0:
        return None
    if seed is not None:
        rng = _random.Random(seed)
    else:
        rng = _random.Random()
    # Configuration model: each vertex gets 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    # Pair up stubs
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # self-loop
        if v in adj[u]:
            return None  # parallel edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    # Verify 3-regular
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    return [list(s) for s in adj], edges

def _improve_girth(adj, n, g, max_iters, rng, deadline):
    """Improve girth via edge swaps (2-switches).
    Returns (adj_list, edge_list) with girth >= g, or None."""
    # Build adjacency list
    adj = [list(s) for s in adj]
    
    def find_short_cycle(adj, limit):
        """Find a cycle shorter than limit, return it or None."""
        for start in range(n):
            # Non-backtracking BFS to find cycles back to start
            visited = {}
            dq = deque()
            dq.append((start, -1, [start]))
            visited[(start, -1)] = True
            while dq:
                v, prev, path = dq.popleft()
                d = len(path)
                if d >= limit:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == start:
                        if d + 1 < limit and d + 1 >= 3:
                            return path + [start]
                    state = (w, v)
                    if state not in visited:
                        visited[state] = True
                        dq.append((w, v, path + [w]))
        return None
    
    def do_swap(adj, e1, e2):
        """Perform a 2-switch: remove edges e1=(a,b), e2=(c,d), add (a,c), (b,d) or (a,d), (b,c).
        e1, e2 are (u,v) tuples. Returns True if swap was valid and applied."""
        a, b = e1
        c, d = e2
        # Check no conflicts
        if a == c or a == d or b == c or b == d:
            return False
        if d in adj[a] or c in adj[b] or d in adj[b] or c in adj[a]:
            return False
        # Remove old edges
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        # Add new edges
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
        return True
    
    def build_edge_list(adj):
        edges = []
        seen = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    
    for it in range(max_iters):
        if _time.time() > deadline:
            return None
        cycle = find_short_cycle(adj, g)
        if cycle is None:
            return adj, build_edge_list(adj)
        # Try to break the cycle via 2-switch
        # The cycle has vertices cycle[0], cycle[1], ..., cycle[-1]=cycle[0]
        # Pick two non-adjacent edges on the cycle and swap them
        cyc_len = len(cycle) - 1  # number of vertices in the cycle
        if cyc_len < 3:
            return None
        # Try multiple random pairs of edges to break
        for _ in range(20):
            # Pick two random edges from the cycle
            idx1 = rng.randint(0, cyc_len - 1)
            idx2 = rng.randint(0, cyc_len - 1)
            if idx1 == idx2 or abs(idx1 - idx2) == 1 or (idx1 + idx2) % cyc_len == 0:
                continue
            e1 = (cycle[idx1], cycle[(idx1 + 1) % cyc_len])
            e2 = (cycle[idx2], cycle[(idx2 + 1) % cyc_len])
            if do_swap(adj, e1, e2):
                break
        # Also try random 2-switches not on the cycle
        if rng.random() < 0.3:
            for _ in range(5):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if len(adj[u]) < 3 or len(adj[v]) < 3:
                    continue
                a = adj[u][rng.randint(0, len(adj[u]) - 1)]
                b = adj[v][rng.randint(0, len(adj[v]) - 1)]
                e1 = (u, a)
                e2 = (v, b)
                if do_swap(adj, e1, e2):
                    break
    
    # Final check
    cycle = find_short_cycle(adj, g)
    if cycle is None:
        return adj, build_edge_list(adj)
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2 ** (g // 2) - 1)
    else:
        moore = 2 * (2 ** (g // 2 - 1) - 1) + 2

    # Base graphs to try (all 3-regular, small, with known structure)
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
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cube_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    prism3_edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5)
    ]

    base_graphs = [
        ("petersen", petersen_edges, 10),
        ("k33", k33_edges, 6),
        ("cube", cube_edges, 8),
        ("k4", k4_edges, 4),
        ("prism3", prism3_edges, 6),
    ]

    best_result = None
    best_n = float('inf')

    # Phase 1: Try voltage lifts with deterministic and random shifts
    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 30:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        for m in range(n_min, n_min + 100):
            if _time.time() > deadline - 25:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # Deterministic shifts
            shifts_det = _quadratic_residues(m, e_count)
            result = _build_lift(base_edges, m, shifts_det)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            shifts_lin = [int((i + 1) * m / (e_count + 1)) for i in range(e_count)]
            result = _build_lift(base_edges, m, shifts_lin)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            shifts_gold = _golden_ratio_shifts(m, e_count)
            result = _build_lift(base_edges, m, shifts_gold)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            for base_mod in [3, 5, 7, 11, 13]:
                shifts_mod = _modular_pattern_shifts(m, e_count, base_mod)
                result = _build_lift(base_edges, m, shifts_mod)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges
                if _time.time() > deadline - 20:
                    break

            # Random shifts
            for attempt in range(30):
                if _time.time() > deadline - 15:
                    break
                seed_val = hash((name, m, attempt, 0))
                rng = _random.Random(seed_val)
                shifts_rand = [rng.randint(1, m - 1) for _ in range(e_count)]
                result = _build_lift(base_edges, m, shifts_rand)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

    # Phase 2: Try random regular graph generation with girth improvement
    # Start from moore bound and go up
    rng_main = _random.Random(42)
    for n_try in range(max(moore, 100), min(moore + 2000, 940), 2):  # n must be even for cubic
        if _time.time() > deadline - 10:
            break
        if n_try >= best_n:
            continue
        
        # Try multiple random seeds
        for seed in range(10):
            if _time.time() > deadline - 5:
                break
            result = _generate_cubic_random(n_try, seed)
            if result is None:
                continue
            adj_list, edge_list = result
            # Check girth
            if _verify_girth_fast(adj_list, n_try, g):
                best_n = n_try
                best_result = (n_try, edge_list)
                return n_try, edge_list
            # Try to improve girth via edge swaps
            improved = _improve_girth(adj_list, n_try, g, max_iters=50, rng=_random.Random(seed + 1000), deadline=deadline - 5)
            if improved is not None:
                adj_imp, edge_imp = improved
                if _verify_girth_fast(adj_imp, n_try, g):
                    best_n = n_try
                    best_result = (n_try, edge_imp)
                    return n_try, edge_imp

    # Phase 3: LUW fallback
    if _time.time() < deadline - 5:
        for d_try in range(6, 14):
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
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    # Final fallback
    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
