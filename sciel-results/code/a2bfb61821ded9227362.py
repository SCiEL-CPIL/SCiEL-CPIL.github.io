# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math as _math
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

def _verify_girth_array(adj, n, g):
    """Faster girth verification using arrays for small graphs."""
    for start in range(n):
        dist = [[-1] * n for _ in range(n)]
        dq = deque()
        dq.append((start, -1))
        dist[start][0] = 0  # dummy, we use (v, prev) pairs via a different encoding
        # Use a dict for (v, prev) -> depth
        dmap = {}
        dmap[(start, -1)] = 0
        dq = deque([(start, -1)])
        while dq:
            v, prev = dq.popleft()
            d = dmap[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                key = (w, v)
                if key not in dmap:
                    dmap[key] = d + 1
                    dq.append((w, v))
    return True

def _base_petersen():
    """Return Petersen graph: 10 vertices, 3-regular, girth 5."""
    # Standard Petersen: outer 5-cycle, inner 5-star, spokes
    edges = []
    # Outer cycle: 0-1-2-3-4-0
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star: 5-7-9-6-8-5 (connecting i+5 to (2*i+5)%5+5)
    for i in range(5):
        edges.append((i + 5, (2 * i + 5) % 5 + 5))
    # Spokes: i to i+5
    for i in range(5):
        edges.append((i, i + 5))
    adj = [[] for _ in range(10)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj, edges

def _base_k33():
    """Return K_{3,3}: 6 vertices, 3-regular, girth 4."""
    edges = []
    for i in range(3):
        for j in range(3):
            edges.append((i, 3 + j))
    adj = [[] for _ in range(6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj, edges

def _base_k4():
    """Return K_4: 4 vertices, 3-regular, girth 3."""
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    adj = [[] for _ in range(4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj, edges

def _base_heawood():
    """Return Heawood graph: 14 vertices, 3-regular, girth 6."""
    # Fano plane complement structure
    # Vertices 0-6: outer, 7-13: inner
    edges = []
    # Outer 7-cycle
    for i in range(7):
        edges.append((i, (i + 1) % 7))
    # Inner: connect i+7 to (2*i+7)%7+7... actually Heawood is bipartite
    # Let me use the standard construction: vertices are 1-subsets and 2-subsets of {1..3}
    # Actually, let me just use a known 3-regular girth-6 graph
    # Heawood: bipartite, parts of 7
    # Part A: 0..6, Part B: 7..13
    # A[i] connected to B[(i+0)%7], B[(i+1)%7], B[(i+3)%7] (one of many)
    # Standard: A[i] to B[i], B[(i+1)%7], B[(i+3)%7]
    for i in range(7):
        edges.append((i, 7 + i % 7))
        edges.append((i, 7 + (i + 1) % 7))
        edges.append((i, 7 + (i + 3) % 7))
    adj = [[] for _ in range(14)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj, edges

def _base_moser_spencer():
    """A 3-regular graph with girth 8: the (3,8)-cage has 30 vertices.
    Let me use a known 3-regular girth-8 graph."""
    # The (3,8)-cage: 30 vertices. Let me construct one.
    # Actually, let me use a simpler approach: take the incidence graph of a known structure
    # For now, use a 3-regular graph I can verify: the 12-vertex graph
    # Let me just use a random 3-regular graph and check girth
    # Better: use the known (3,8)-cage structure
    # The (3,8)-cage has 30 vertices. I'll use a known edge list.
    # Actually, let me construct a 3-regular girth-7 graph (14 vertices = Heawood)
    # and a 3-regular girth-8 graph.
    # For girth 8, I need 30 vertices minimum.
    # Let me use a different approach: just use Petersen (girth 5) and K3,3 (girth 4)
    # and rely on the lift to increase girth.
    return None

def _voltage_lift(adj_base, edges_base, s, voltages, deadline=None):
    """Build the voltage lift of adj_base with voltage assignment voltages.
    voltages is a dict mapping edge (u,v) with u<v to an element of Z_s.
    Returns (n, adj, edges) or None if not regular."""
    m = len(adj_base)
    n = m * s
    adj = [[] for _ in range(n)]
    edge_list = []
    edge_set = set()
    
    for u, v in edges_base:
        key = (min(u, v), max(u, v))
        if key not in voltages:
            voltages[key] = 0
        volt = voltages[key]
        # For each lift level a in Z_s:
        # edge from (u, a) to (v, a + volt)
        # and from (v, a) to (u, a - volt)
        for a in range(s):
            u_lift = u * s + a
            v_lift = v * s + ((a + volt) % s)
            if u_lift != v_lift:
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
                ek = (min(u_lift, v_lift), max(u_lift, v_lift))
                if ek not in edge_set:
                    edge_set.add(ek)
                    edge_list.append(ek)
                else:
                    # Multi-edge, not simple
                    return None
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edge_list

def _find_lift(adj_base, edges_base, g, deadline):
    """Search for a voltage lift of adj_base with girth >= g.
    Returns (n, edges) for the smallest valid lift found."""
    m = len(adj_base)
    # The girth of the lift is at least girth(base) + (s-1)*something
    # For a base with girth g_b, the lift with s levels has girth >= g_b + (s-1)*g_min_cycle
    # where g_min_cycle relates to the voltage structure
    # Heuristic: try s from small to large
    best_n = float('inf')
    best_edges = None
    
    # Try various lift sizes
    # For girth g, we need roughly s >= g / girth_base + 1
    # But this is very rough. Let's try a range.
    max_s = min(200, max(5, g + 10))
    
    for s in range(2, max_s + 1):
        if _time.time() > deadline - 5:
            break
        n_try = m * s
        if n_try >= best_n:
            break
        
        # Try multiple random voltage assignments
        n_attempts = 200 if s <= 50 else 100
        for attempt in range(n_attempts):
            if _time.time() > deadline - 5:
                break
            rng = _random.Random(attempt * 31 + s * 17 + m)
            voltages = {}
            for u, v in edges_base:
                key = (min(u, v), max(u, v))
                voltages[key] = rng.randint(0, s - 1)
            
            result = _voltage_lift(adj_base, edges_base, s, voltages)
            if result is None:
                continue
            n, adj, edges = result
            if n >= best_n:
                continue
            if _verify_girth_fast(adj, n, g):
                best_n = n
                best_edges = edges
                break  # Found one for this s, try smaller n next
    
    if best_edges is not None:
        return best_n, best_edges
    return None

def _deterministic_voltage_lift(adj_base, edges_base, s, g, deadline):
    """Try deterministic voltage assignments based on primitive roots."""
    m = len(adj_base)
    best_n = float('inf')
    best_edges = None
    
    # Use voltage = k * edge_index mod s for various k
    for k in range(1, min(s, 50)):
        if _time.time() > deadline - 5:
            break
        voltages = {}
        for idx, (u, v) in enumerate(edges_base):
            key = (min(u, v), max(u, v))
            voltages[key] = (k * idx) % s
        
        result = _voltage_lift(adj_base, edges_base, s, voltages)
        if result is None:
            continue
        n, adj, edges = result
        if n >= best_n:
            continue
        if _verify_girth_fast(adj, n, g):
            best_n = n
            best_edges = edges
            break
    
    if best_edges is not None:
        return best_n, best_edges
    return None

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graph of girth >= g."""
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

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140
    
    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges
    
    best_n = float('inf')
    best_edges = None
    
    # Strategy 1: Voltage lifts over Petersen (girth 5)
    pet_adj, pet_edges = _base_petersen()
    result = _find_lift(pet_adj, pet_edges, g, deadline - 10)
    if result is not None:
        n, edges = result
        if n < best_n:
            best_n = n
            best_edges = edges
    
    # Strategy 2: Voltage lifts over K_{3,3} (girth 4)
    k33_adj, k33_edges = _base_k33()
    result = _find_lift(k33_adj, k33_edges, g, deadline - 10)
    if result is not None:
        n, edges = result
        if n < best_n:
            best_n = n
            best_edges = edges
    
    # Strategy 3: Voltage lifts over Heawood (girth 6)
    hw_adj, hw_edges = _base_heawood()
    result = _find_lift(hw_adj, hw_edges, g, deadline - 10)
    if result is not None:
        n, edges = result
        if n < best_n:
            best_n = n
            best_edges = edges
    
    # Strategy 4: Deterministic lifts with more attempts
    if _time.time() < deadline - 10:
        for base_adj, base_edges, name in [(pet_adj, pet_edges, 'pet'), (k33_adj, k33_edges, 'k33'), (hw_adj, hw_edges, 'hw')]:
            if _time.time() > deadline - 10:
                break
            for s in range(2, 100):
                if _time.time() > deadline - 10:
                    break
                n_try = len(base_adj) * s
                if n_try >= best_n:
                    break
                result = _deterministic_voltage_lift(base_adj, base_edges, s, g, deadline - 10)
                if result is not None:
                    n, edges = result
                    if n < best_n:
                        best_n = n
                        best_edges = edges
    
    # Strategy 5: LUW fallback
    if _time.time() < deadline - 5:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_edges = edges_luw
    
    if best_edges is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_edges = edges_luw
        best_n = n_luw
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
