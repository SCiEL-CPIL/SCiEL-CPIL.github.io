"""
Combined program: Algebraic voltage lifts (primitive roots) + LUW fallback.
Strategy:
1. Use the Petersen graph as a base for voltage lifts.
2. Assign voltages using powers of a primitive root modulo m to maximize
   the 'spread' of voltage sums, which helps avoid short cycles.
3. Scan m from the Moore bound upwards.
4. Use fast non-backtracking BFS to verify girth.
5. Fallback to LUW construction if the lift search is too slow or fails.
"""
# EVOLVE-BLOCK-START
import time as _time
import math
from collections import deque

def _is_primitive_root(a, m):
    """Check if a is a primitive root modulo m."""
    if m == 1:
        return True
    if m == 2:
        return a % 2 == 1
    # Factorize m-1 to check orders
    def factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            else:
                d += 1
        if n > 1:
            factors.append(n)
        return factors

    phi = m - 1
    if m == 2:
        phi = 1
    else:
        # Calculate phi(m)
        n = m
        phi = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                phi -= phi // p
            p += 1
        if n > 1:
            phi -= phi // n

    # If a is not coprime to m, it's not a primitive root
    if math.gcd(a, m) != 1:
        return False

    factors = factorize(phi)
    for p in factors:
        if pow(a, phi // p, m) == 1:
            return False
    return True

def _get_primitive_root(m):
    """Find a primitive root modulo m. Returns None if m=1 or 2."""
    if m <= 2:
        return 0
    for a in range(2, m):
        if _is_primitive_root(a, m):
            return a
    return None

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # For girth g, we need to ensure no cycle of length < g.
    # A non-backtracking walk of length L starting and ending at v
    # implies a cycle of length <= L.
    # We stop expanding if depth >= g-1.
    for start in range(n):
        # dist[(v, prev)] = depth
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
    """
    Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid.
    """
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
    # Check regularity
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
        # Heuristic for d
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y:(x+y)%q
        sub = lambda x,y:(x-y)%q
        mul = lambda x,y:x*y%q
        
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
    start_time = _time.time()
    deadline = start_time + 140  # Safety margin

    if k != 3:
        # LUW fallback for other k
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Base graph: Petersen
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    e_count = len(petersen_edges)

    best_n = float('inf')
    best_result = None

    # Scan m from Moore bound
    # Moore bound for g=16 is 510.
    # We expect the lift to be valid for m around 50-100 if the voltage assignment is good.
    # But we must start from a reasonable lower bound.
    # Actually, the lift size is n_base * m.
    # We want n_base * m to be small.
    # So we should scan m from 1 upwards?
    # No, if m is small, the voltages might collide or form short cycles.
    # Let's scan m from 1 to, say, 200.
    # If we find a valid graph with n < best_n, update.
    
    # To be safe and fast, we'll try m in increasing order.
    # We stop if n_base * m exceeds the best known LUW result or if time runs out.
    
    # Get LUW result first to have an upper bound
    # LUW with d=8 gives 2*3^8 = 13122 vertices.
    # LUW with d=7 might work? 2*3^7 = 4374.
    # Let's compute LUW for d=8 as a benchmark.
    try:
        n_luw_bench, edges_luw_bench = _luw_construct(3, g, d_override=8)
        best_n = n_luw_bench
        best_result = (n_luw_bench, edges_luw_bench)
    except Exception:
        best_n = 20000 # Fallback upper bound

    # Try algebraic lifts
    for m in range(1, 200):
        if _time.time() > deadline - 10:
            break
        n = n_base * m
        if n >= best_n:
            break # No point in trying larger m if we already have a smaller result

        # Find primitive root
        r = _get_primitive_root(m)
        if r is None:
            # If no primitive root, use a random spread or skip
            # For m=1, shifts are all 0.
            if m == 1:
                shifts = [0] * e_count
            else:
                # Use a simple linear spread as fallback
                shifts = [(i * (m // e_count)) % m for i in range(e_count)]
        else:
            # Assign voltages as powers of r
            # v_i = r^(i+1) mod m ? Or r^i?
            # If r is primitive root, r^i cycles through all units.
            # We want the sums to be non-zero.
            shifts = [pow(r, i + 1, m) for i in range(e_count)]
            # Ensure shifts are in 0..m-1
            # pow already returns mod m

        result = _build_lift(petersen_edges, m, shifts)
        if result is not None:
            n_r, adj, edges = result
            if _verify_girth_fast(adj, n_r, g):
                best_n = n_r
                best_result = (n_r, edges)
                # If we found a very small graph, we can return early
                # Moore bound is 510. If n_r < 600, it's excellent.
                if n_r < moore + 50:
                    return n_r, edges

    # If we didn't find a better graph, try a few random shifts for the best m found?
    # Or just return the best result (likely LUW or the smallest lift found)
    
    if best_result is not None:
        return best_result

    # Final fallback
    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
