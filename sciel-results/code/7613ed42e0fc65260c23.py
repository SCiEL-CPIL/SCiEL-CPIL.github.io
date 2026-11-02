import time
import random
import numpy as np
from itertools import combinations_with_replacement, product as iproduct
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Moore bound for k=3, g=16: 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*255 = 766
    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
    n_start = moore_bound if moore_bound % 2 == 0 else moore_bound + 1

    best_n = None
    best_edges = None

    # Phase 1: Systematic circulant search - try many a values for each n
    # For even n, connection set is {a, n-a, n/2}
    # Key insight: we want a with large multiplicative order mod n to maximize girth
    phase1_deadline = start_time + 50

    # Try n values from moore_bound upward
    for n in range(n_start, min(n_start + 600, 2000)):
        if n % 2 != 0:
            continue
        if time.time() > phase1_deadline:
            break

        half = n // 2

        # Generate candidate a values sorted by heuristic: prefer a with gcd(a,n)=1
        # and large order
        candidates_a = []
        for a in range(1, half):
            candidates_a.append(a)

        # Sort by: prefer coprime with n, then by multiplicative order (larger = better)
        def a_heuristic(a):
            import math
            gcd_val = math.gcd(a, n)
            if gcd_val > 1:
                # Not coprime, less likely to give good girth
                return (1, -a)  # lower priority
            else:
                return (0, -a)  # higher priority, larger a first
        
        candidates_a.sort(key=a_heuristic)

        # Try top candidates first (limit per n to manage time)
        max_a_per_n = 50
        tried = 0
        for a in candidates_a:
            if tried >= max_a_per_n:
                break
            if time.time() > phase1_deadline:
                break
            tried += 1

            # Quick rejection: if a and half have small order, skip
            # Compute order of a mod n
            # But just check girth directly
            girth = _circulant_girth(n, a, half)
            if girth >= g:
                edges = _circulant_edges(n, a, half)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid a for this n, move to next n

    if best_n is not None:
        return best_n, best_edges

    # Phase 2: Exhaustive circulant search for smaller range
    phase2_deadline = start_time + 90
    for n in range(n_start, min(n_start + 300, 1500)):
        if n % 2 != 0:
            continue
        if time.time() > phase2_deadline:
            break
        half = n // 2
        for a in range(1, half):
            if time.time() > phase2_deadline:
                break
            girth = _circulant_girth(n, a, half)
            if girth >= g:
                edges = _circulant_edges(n, a, half)
                return n, edges

    # Phase 3: Random voltage/lifting approach
    # Start with a small base and try random 3-regular graphs
    phase3_deadline = start_time + 110
    best_rand_n = None
    best_rand_edges = None

    # Try random regular graphs with n vertices
    for n_try in range(n_start, min(n_start + 200, 1200)):
        if n_try % 2 != 0:
            continue
        if time.time() > phase3_deadline:
            break

        # Try a few random graphs
        for attempt in range(5):
            if time.time() > phase3_deadline:
                break
            edges = _generate_random_regular_graph(n_try, 3)
            if edges is not None:
                if _check_girth_fast(edges, n_try, g):
                    if best_rand_n is None or n_try < best_rand_n:
                        best_rand_n = n_try
                        best_rand_edges = edges
                    break  # Found valid graph for this n, try smaller n

    if best_rand_n is not None:
        return best_rand_n, best_rand_edges

    # Phase 4: Fallback to LUW
    return _luw_construct(k, g)

def _circulant_girth(n, a, half):
    """Compute the girth of circulant graph on Z_n with steps {a, -a, half}.
    Uses non-backtracking BFS. Returns the girth (shortest cycle length).
    Optimized for speed.
    """
    steps = [a, (-a) % n, half]
    reverse = {0: 1, 1: 0, 2: 2}
    
    for start_step in range(3):
        # BFS: state = (position, last_step), distance = number of steps taken
        visited = {}  # (pos, last_step) -> distance
        queue = deque()
        pos = steps[start_step] % n
        visited[(pos, start_step)] = 1
        queue.append((pos, start_step, 1))
        
        while queue:
            pos, last, dist = queue.popleft()
            if dist >= 20:  # Cap: we only care about girth < 20
                continue
            for s in range(3):
                if s == reverse[last]:
                    continue  # non-backtracking
                new_pos = (pos + steps[s]) % n
                new_state = (new_pos, s)
                if new_pos == 0:
                    # We've returned to start!
                    if s != reverse[start_step]:
                        return dist + 1
                if new_state in visited:
                    cycle_len = dist + 1 - visited[new_state]
                    if cycle_len > 1:
                        return cycle_len
                else:
                    visited[new_state] = dist + 1
                    queue.append((new_pos, s, dist + 1))
    
    return 100  # No short cycle found

def _circulant_edges(n, a, half):
    """Generate edges for circulant graph on Z_n with steps {a, -a, half}."""
    edges = []
    a_mod = a % n
    for v in range(n):
        # Edge to v + a (only add if v < v+a to avoid duplicates)
        u = (v + a_mod) % n
        if v < u:
            edges.append((v, u))
        # Edge to v - a
        u = (v - a_mod) % n
        if v < u:
            edges.append((v, u))
        # Edge to v + half
        u = (v + half) % n
        if v < u:
            edges.append((v, u))
    return edges

def _luw_construct(k, g):
    # Original code
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

def _luw_construct(k, g):
    # Original code
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

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph using configuration model with retries."""
    if n * k % 2 != 0 or n < k:
        return None
    
    max_retries = 30
    for _ in range(max_retries):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

def _check_girth_fast(edges, n, g):
    """Fast girth check using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    depth_limit = g - 1
    
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            state = (start, nb)
            visited[state] = 1
            queue.append((start, nb, 1))
        
        while queue:
            u, prev, depth = queue.popleft()
            if depth >= depth_limit:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                new_state = (u, v)
                if new_state in visited:
                    cycle_len = depth - visited[new_state]
                    if cycle_len < g:
                        return False
                else:
                    visited[new_state] = depth + 1
                    queue.append((u, v, depth + 1))
    return True

def _improve_girth(edges, n, g, max_iter, time_limit):
    return None
