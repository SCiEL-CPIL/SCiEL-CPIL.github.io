import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = float('inf')
    best_edges = None

    if k == 3 and g == 16:
        # Try d=6 (n=1458) with random rules
        # This is the most promising area for improvement over the seed (which likely uses d=7+)
        
        # First, try standard rules for d=6, 7, 8 to have a baseline
        for d in [6, 7, 8]:
            if time.time() > deadline - 30:
                break
            try:
                n_luw, edges_luw = _construct_luw_standard(k, d)
                if n_luw < best_n:
                    if _has_girth_at_least_fast(n_luw, edges_luw, g):
                        best_n = n_luw
                        best_edges = edges_luw
            except Exception:
                continue
        
        # If we haven't found a small graph, try random rules for d=6
        # We have time to try several random configurations
        if best_n > 1458: # If standard d=6 failed or wasn't tried/failed girth
            random.seed(42)
            attempts = 0
            while time.time() < deadline - 10 and attempts < 50:
                attempts += 1
                try:
                    n_luw, edges_luw = _construct_luw_random_rules(k, 6, random)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if best_n == 1458:
                                break # Can't do better with d=6
                except Exception:
                    continue
        
        # If still large, try d=7 with random rules to ensure validity if standard failed
        if best_n > 4374:
             random.seed(43)
             attempts = 0
             while time.time() < deadline - 5 and attempts < 20:
                attempts += 1
                try:
                    n_luw, edges_luw = _construct_luw_random_rules(k, 7, random)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                except Exception:
                    continue

        if best_n != float('inf'):
            return best_n, best_edges

    # Generic fallback
    # Try standard LUW with increasing d
    min_d = max(3, g // 2 - 2)
    max_d = g + 5
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        try:
            n_luw, edges_luw = _construct_luw_standard(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
                
    # Last resort - use a larger d to ensure validity
    n_luw, edges_luw = _construct_luw_standard(k, max_d + 3)
    return n_luw, edges_luw

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # If we see a visited node that is not the parent, 
                    # it's a cycle.
                    # The length of the cycle is dist[u] + dist[v] + 1
                    # But we must ensure u and v are not connected via the tree path 
                    # in a way that makes this a "back edge" to an ancestor 
                    # which would imply a shorter cycle if dist difference is small.
                    # Actually, in BFS, if v is visited and v != parent[u], 
                    # the cycle length is dist[u] + dist[v] + 1.
                    # We need this to be >= g.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_standard(k, d):
    """
    Standard LUW construction for 3-regular graphs.
    Uses the standard LUW rules for k=3.
    """
    q = k  # For k=3, q=3
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Standard LUW rules for k=3
    # rule[c] = (a, b) where y[c] = y[a] * x[b] (or x[a] * y[b] for side 1)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    return _luw_build(k, d, rule, q)

def _construct_luw_random_rules(k, d, rng):
    """
    LUW construction with random linear rules.
    """
    q = k
    
    # Generate random rules
    # y[c] = a*y[c-1] + b*x[c] + c*x[c-1] ...
    # Simplified: y[c] = a*y[c-1] + b*x[c]
    rule = {}
    for c in range(2, d + 1):
        a = rng.randint(0, q - 1)
        b = rng.randint(1, q - 1) # b != 0 to ensure mixing
        rule[c] = (a, b)

    return _luw_build_random(k, d, rule, q)

def _luw_build(k, d, rule, q):
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _luw_build_random(k, d, rule, q):
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = a * y[c-1] + b * x[c]
                    y[c] = add(mul(a, y[c-1]), mul(b, x[c]))
                else:
                    # y[c] = a * x[c-1] + b * y[c]  (symmetric?)
                    # Let's stick to a similar structure
                    y[c] = sub(mul(a, x[c-1]), mul(b, y[c-1]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
