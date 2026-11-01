"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
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

def _verify_girth_array(adj_list, n, g):
    """Fast verification using flat array distance tracking for k=3 graphs."""
    if g < 3:
        return True
    # For k-regular graph, state is (v, prev_edge_idx) where prev_edge_idx is 0..k-1
    # We use a flat array indexed by v * k + prev_idx
    k = 3
    # dist[v][prev_idx] = distance, -1 means unvisited
    # Using a list of arrays for speed
    dist = [[-1] * k for _ in range(n)]
    dq = deque()
    
    for start in range(n):
        # Reset dist - use a generation counter to avoid reallocation
        # Actually for small n, just reset
        for i in range(n):
            dist[i][0] = -1
            dist[i][1] = -1
            dist[i][2] = -1
        
        # Start: from vertex start, no previous edge
        # We need to handle the "no previous" case specially
        # State: (vertex, prev_vertex) but we encode prev_vertex as index
        # For k=3, we can encode state as (v, prev_idx) where prev_idx is 0,1,2 or 3 (for none)
        
        # Use a different approach: dist[v][prev_idx] where prev_idx in 0..2
        # Initial state: at start, coming from "nowhere" - we try all 3 neighbors
        # Mark (start, -1) as visited at distance 0
        # We'll use a separate flag or just initialize by pushing all neighbors
        
        # Push initial neighbors
        for idx, w in enumerate(adj_list[start]):
            # State: (w, start), meaning at w, came from start
            # prev_idx for w: which index in adj_list[w] is start?
            # We need to find the index of 'start' in adj_list[w]
            # Precompute this
            pass
        
        # Better: precompute for each vertex, a mapping from neighbor to its index
        # Let's restructure: use (v, prev_v) pairs but with array
        # State space: n * (k+1) where prev can be 0..k-1 or "none"
        # Use prev = -1 encoded as k (index k)
        
        # Actually, let's use a 2D array: dist[v][prev_idx] where prev_idx is 0..k
        # prev_idx = k means "no previous vertex" (only valid for start)
        # But this is complex. Let's use a simpler but still fast approach.
        
        # Use a dictionary but with integer keys for speed
        visited = set()
        dq.clear()
        dq.append((start, -1, 0))
        visited.add(start * k)  # encode (start, -1) as start*k + 0 (special)
        
        while dq:
            v, prev, d = dq.popleft()
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                # Encode state (w, v) as w * k + index_of_v_in_adj[w]
                # Need index of v in adj_list[w]
                # For k=3, we can search manually
                aw = adj_list[w]
                if aw[0] == v:
                    enc = w * 3 + 0
                elif aw[1] == v:
                    enc = w * 3 + 1
                else:
                    enc = w * 3 + 2
                if enc not in visited:
                    visited.add(enc)
                    dq.append((w, v, d + 1))
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
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

def _get_base_graphs():
    """Return a list of base 3-regular graphs as (edges, n_base)."""
    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    # K_{3,3}
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    # Cube graph (Q3) - 8 vertices, girth 4
    cube_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5), (1, 4),
        (2, 3), (2, 6), (2, 4),
        (3, 7), (3, 6), (3, 5),
        (4, 5), (4, 6), (4, 7),
        (5, 7), (6, 7)
    ]
    # Wait, that's not right. Q3 has 12 edges.
    # Vertices: 000,001,010,011,100,101,110,111
    # 0: 000, 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
    cube_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5), (1, 4),
        (2, 3), (2, 6), (2, 4),
        (3, 7), (3, 6), (3, 5),
        (4, 5), (4, 6), (4, 7),
        (5, 7), (6, 7)
    ]
    # That's 18 edges, but Q3 should have 12. Let me redo.
    # 000-001, 000-010, 000-100
    # 001-011, 001-101
    # 010-011, 010-110
    # 011-111
    # 100-101, 100-110
    # 101-111
    # 110-111
    cube_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    # That's 12 edges. Good.
    
    # Also add the 3-prism (two triangles with matching edges) - but that has triangles.
    # The dodecahedron graph? 20 vertices, girth 5. Too large for a base.
    
    # Let's add a 12-vertex 3-regular graph: the utility graph K3,3 is 6, Petersen is 10.
    # The 3-cube Q3 is 8 vertices, girth 4.
    
    return [(petersen_edges, 10), (k33_edges, 6), (cube_edges, 8)]

def _build_lift(base_edges, n_base, m, voltages, g):
    """Build a voltage graph lift. Returns (n, adj_list, edges) or None if invalid."""
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
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
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, adj_list, edges

def _find_primitive_root(m):
    """Find a primitive root modulo m (requires m prime). Returns None otherwise."""
    if m <= 1:
        return None
    import math
    if m < 2:
        return None
    for i in range(2, int(math.isqrt(m)) + 1):
        if m % i == 0:
            return None
    phi = m - 1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    for g in range(2, m):
        is_primitive = True
        for p in factors:
            if pow(g, phi // p, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return None

def _voltage_lift_search(base_edges, n_base, m, g, seed, n_attempts=50, deadline=None):
    """Try multiple random voltage assignments on a base graph lift.
    Returns (n, edges) if a valid girth-g graph is found, else None."""
    rng = _random.Random(seed)
    n_lift = n_base * m
    
    for attempt in range(n_attempts):
        if deadline is not None and _time.time() > deadline:
            break
        voltages = {}
        for u, v in base_edges:
            vol = rng.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    return None

def _deterministic_voltage_lift(base_edges, n_base, m, g):
    """Use deterministic voltage assignment based on primitive root powers."""
    root = _find_primitive_root(m)
    if root is None:
        return None
    
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        voltages[(u, v)] = (root ** i) % m
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift(base_edges, n_base, m, voltages, g)
    if result is None:
        return None
    n, adj_list, edges = result
    if _verify_girth_array(adj_list, n, g):
        return n, edges
    return None

def _random_3reg(n, g, seed):
    """Generate a random 3-regular graph and check girth."""
    rng = _random.Random(seed)
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
    adj_list = [sorted(adj[i]) for i in range(n)]
    if _verify_girth_array(adj_list, n, g):
        return n, edges
    return None

def _cycle_avoiding_build(n, g, seed, deadline=None):
    """Build a 3-regular graph by adding vertices one at a time,
    choosing neighbors that avoid creating cycles shorter than g.
    Finish with a random matching of dangling stubs."""
    rng = _random.Random(seed)
    max_depth = g - 1  # max non-backtracking walk length to check
    
    # adj[v] = list of neighbors (up to 3)
    adj = [[] for _ in range(n)]
    edges = []
    
    # For each vertex, track how many edges it has
    degree = [0] * n
    
    # BFS from a start vertex to find all vertices within distance < g/2
    # that would create a short cycle if connected
    def _would_create_short_cycle(u, v, adj, degree, max_d):
        """Check if adding edge (u,v) creates a cycle shorter than g.
        Uses non-backtracking BFS from u (not going through v) up to depth max_d-1,
        and from v (not going through u) up to depth max_d-1.
        If the two frontiers meet, there's a cycle of length <= max_d.
        Actually: if u and v are already connected by a non-backtracking path
        of length <= max_d-1 (not using the new edge), then adding (u,v)
        creates a cycle of length <= max_d.
        
        More precisely: cycle length = path_length + 1. We need path_length + 1 < g,
        i.e., path_length < g-1, i.e., path_length <= g-2.
        So we check if there's a non-backtracking path from u to v of length <= g-2
        that doesn't use the edge (u,v) (which doesn't exist yet).
        
        BFS from u, not backtracking, up to depth g-2. If we reach v, cycle exists.
        """
        # Non-backtracking BFS from u, depth up to max_d
        # State: (current_vertex, prev_vertex), distance
        # Start: all neighbors of u (not v), depth 1
        visited = set()
        dq = deque()
        for w in adj[u]:
            if w == v:
                continue
            visited.add((w, u))
            dq.append((w, u, 1))
        while dq:
            cur, prev, d = dq.popleft()
            if cur == v:
                return True
            if d >= max_d:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                state = (nxt, cur)
                if state not in visited:
                    visited.add(state)
                    dq.append((nxt, cur, d + 1))
        return False
    
    # Build the graph
    # Start with vertex 0
    adj[0] = []
    
    # We'll add vertices 1..n-1
    # Each new vertex v connects to 2 existing vertices (or 1 existing + 1 future)
    # Actually, let's use a different approach:
    # Start with a tree-like structure, then add edges to make it 3-regular
    
    # Better approach: 
    # 1. Build a spanning tree (or forest) first
    # 2. Add remaining edges avoiding short cycles
    
    # Simplest: use configuration model but with cycle checking
    # Stubs: each vertex has 3 stubs
    # Process stubs one by one, matching them while avoiding short cycles
    
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    # For efficiency, maintain a set of "active" stubs
    # and for each vertex, its current neighbors
    
    # This is O(n^2) in worst case for cycle checking, which is too slow for large n.
    # For n up to ~2000 and g=16, max_d=15, BFS is bounded by 3*2^14 ~ 49152 states.
    # But doing this for each of n*3/2 ~ 3000 edges... 3000 * 50000 = 150M ops. Too slow.
    
    # Alternative: only check cycles for the last few edges added,
    # and do a full verification at the end.
    
    # Actually, let's just do the full random 3-reg and verify.
    # The cycle-avoiding approach is too slow per-edge.
    
    # Instead: use a smarter random construction.
    # 1. Create a random 3-regular graph
    # 2. If it has short cycles, try to fix them by edge swaps
    
    return None  # Fall through to other methods

def _structured_voltage_lift(base_edges, n_base, m, g, seed, deadline=None):
    """Try structured voltage assignments: quadratic residues, linear, etc."""
    rng = _random.Random(seed)
    n_lift = n_base * m
    n_edges = len(base_edges)
    
    # Strategy: use powers of a base element
    for base_elem in range(2, min(m, 30) + 1):
        if deadline is not None and _time.time() > deadline:
            break
        voltages = {}
        for i, (u, v) in enumerate(base_edges):
            vol = (base_elem ** i) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    # Strategy: linear voltage assignment vol_e = a*e + b mod m
    for a in range(1, min(m, 20)):
        if deadline is not None and _time.time() > deadline:
            break
        for b in range(0, min(m, 10)):
            voltages = {}
            for i, (u, v) in enumerate(base_edges):
                vol = (a * i + b) % m
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
            result = _build_lift(base_edges, n_base, m, voltages, g)
            if result is None:
                continue
            n, adj_list, edges = result
            if _verify_girth_array(adj_list, n, g):
                return n, edges
    
    # Strategy: quadratic voltage vol_e = a*e^2 + b*e + c mod m
    for a in range(1, min(m, 10)):
        if deadline is not None and _time.time() > deadline:
            break
        for b in range(0, min(m, 5)):
            for c in range(0, min(m, 5)):
                voltages = {}
                for i, (u, v) in enumerate(base_edges):
                    vol = (a * i * i + b * i + c) % m
                    voltages[(u, v)] = vol
                    voltages[(v, u)] = (-vol) % m
                result = _build_lift(base_edges, n_base, m, voltages, g)
                if result is None:
                    continue
                n, adj_list, edges = result
                if _verify_girth_array(adj_list, n, g):
                    return n, edges
    
    # Strategy: random but with structure
    for attempt in range(50):
        if deadline is not None and _time.time() > deadline:
            break
        voltages = {}
        perms = list(range(m))
        rng.shuffle(perms)
        for i, (u, v) in enumerate(base_edges):
            vol = perms[i % m]
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Get base graphs
    base_graphs = _get_base_graphs()
    
    # Strategy 1: Voltage lifts - try many m values with multiple strategies
    # Allocate 70% of time to voltage lifts
    vl_deadline = start_time + (deadline - start_time) * 0.70
    
    for base_edges, n_base in base_graphs:
        if _time.time() > vl_deadline:
            break
        m_min = max(1, (moore + n_base - 1) // n_base)
        # Allow more m values to explore
        m_max = m_min + 1200
        
        for m in range(m_min, m_max):
            if _time.time() > vl_deadline:
                break
            n_lift = n_base * m
            if n_lift < moore:
                continue
            if n_lift >= best_n:
                break
            
            # Try deterministic voltage first (fast)
            result = _deterministic_voltage_lift(base_edges, n_base, m, g)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
            
            # Try structured voltage (medium)
            result = _structured_voltage_lift(base_edges, n_base, m, g, seed=m * 7919 + n_base * 104729, deadline=vl_deadline)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
            
            # Try random voltage assignments (slower)
            result = _voltage_lift_search(base_edges, n_base, m, g, seed=m * 100000 + n_base, n_attempts=120, deadline=vl_deadline)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
    
    # Strategy 2: LUW construction as fallback
    if _time.time() < deadline - 15:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: Random 3-regular graphs (less promising but cheap to try)
    if _time.time() < deadline - 10 and best_result is not None:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < best_n and _time.time() < deadline - 5:
            for attempt in range(15):
                if _time.time() > deadline - 5:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 104729 + n_try)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 24
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
