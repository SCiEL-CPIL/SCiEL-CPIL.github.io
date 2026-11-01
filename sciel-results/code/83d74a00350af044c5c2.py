"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
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

def _circulant_3reg(n, a, b):
    """Build a 3-regular circulant graph on Z_n with offsets ±a, ±b.
    Returns (n, adj, edges) or None if not simple/3-regular."""
    if a == 0 or b == 0:
        return None
    if a == b or a == -b or a == 0 or b == 0:
        return None
    if n % 2 == 0 and a == b:
        return None
    # Check offsets are distinct mod n
    offsets = [a, -a, b, -b]
    offsets_mod = [x % n for x in offsets]
    if len(set(offsets_mod)) < 4:
        return None
    # Also check a != b mod n, a != -b mod n, etc.
    if a % n == b % n or a % n == (-b) % n or (-a) % n == b % n or (-a) % n == (-b) % n:
        return None
    adj = [set() for _ in range(n)]
    edges = []
    for v in range(n):
        for off in [a, b]:
            w = (v + off) % n
            if w not in adj[v]:
                adj[v].add(w)
                adj[w].add(v)
                if v < w:
                    edges.append((v, w))
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    return n, adj, edges

def _search_circulant(n, g, seed, max_attempts=5000):
    """Search for valid (a,b) offsets for circulant 3-regular graph of girth >= g."""
    rng = _random.Random(seed)
    # Try structured offsets first: golden ratio, primes, etc.
    import math
    phi = (1 + 5**0.5) / 2
    # Candidate a values: spread out
    a_candidates = set()
    for i in range(1, n // 2 + 1):
        a_candidates.add(i)
    # Sort by "spread" quality - prefer values near n/4, n/3, etc.
    # Actually just random search with some structure
    attempts = 0
    for attempt in range(max_attempts):
        if attempt < 200:
            # Structured: use golden ratio related offsets
            a = int(n * phi * (attempt + 1) / (max_attempts * 0.2)) % n
            if a == 0 or a == n - a:
                a = (a + 1) % n
            b = int(n * phi * (attempt + 101) / (max_attempts * 0.2)) % n
            if b == 0 or b == n - b:
                b = (b + 1) % n
        else:
            a = rng.randint(1, n - 1)
            b = rng.randint(1, n - 1)
        
        result = _circulant_3reg(n, a, b)
        if result is None:
            continue
        n_c, adj, edges = result
        if _verify_girth_fast(adj, n_c, g):
            return n_c, edges
        attempts += 1
    return None

def _luw_construct(q, g):
    """LUW construction as fallback. Builds (q+1)-regular graph of girth >= g."""
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
    
    # Get LUW baseline first (it's fast and gives a good upper bound)
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Strategy 1: Search circulant graphs (Z_n with offsets ±a, ±b)
    # Try n values from Moore bound up to LUW size
    n_max = min(best_n, 10000)
    
    for n_try in range(moore, n_max + 1, 2):
        if _time.time() > deadline - 15:
            break
        if n_try >= best_n:
            break
        # For circulant 3-regular, n must be even (for 4 distinct offsets)
        # unless we use a different structure
        if n_try % 2 != 0:
            continue
        
        result = _search_circulant(n_try, g, seed=n_try * 31 + 7, max_attempts=2000)
        if result is not None:
            n_c, edges_c = result
            if n_c < best_n:
                best_n = n_c
                best_result = (n_c, edges_c)
    
    # Strategy 2: Try more n values with more attempts if time permits
    if _time.time() < deadline - 15 and best_n < n_luw:
        for n_try in range(moore, min(best_n, n_max), 2):
            if _time.time() > deadline - 10:
                break
            if n_try >= best_n:
                break
            if n_try % 2 != 0:
                continue
            result = _search_circulant(n_try, g, seed=n_try * 7919 + 13, max_attempts=5000)
            if result is not None:
                n_c, edges_c = result
                if n_c < best_n:
                    best_n = n_c
                    best_result = (n_c, edges_c)
    
    # Fallback: LUW construction
    if best_result is None:
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
