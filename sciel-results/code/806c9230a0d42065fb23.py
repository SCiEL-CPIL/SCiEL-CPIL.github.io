import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate the LUW graph as the primary construction
    n_luw, edges_luw = _construct_luw(k, g, deadline)
    
    best_n = n_luw
    best_edges = edges_luw
    
    # If we have significant time remaining, try a limited search for smaller graphs
    elapsed = time.time() - start_time
    if elapsed < 90 and best_n > 100:  # Only search if LUW is large and we have time
        # Moore bound for 3-regular graph of girth g
        if g % 2 == 0:
            half_g = g // 2
            moore_bound = 2 * (1 + k * sum([(k-1)**i for i in range(half_g-1)]))
        else:
            half_g = g // 2
            moore_bound = 1 + k * sum([(k-1)**i for i in range(half_g)])
        
        # For k=3, g=16: moore_bound = 2*(1 + 3*(1+2+4+8+16+32+64)) = 2*(1+3*127) = 2*382 = 764
        # But the LUW graph should be smaller than the Moore bound
        # Try a few random graphs of size close to moore_bound if time permits
        remaining = deadline - time.time()
        if remaining > 30:
            # Try at most 5 random graphs, checking girth quickly
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                # Try a size slightly above moore bound
                n_try = moore_bound + 2 * attempt
                if n_try % 2 != 0:
                    n_try += 1
                if n_try >= best_n:
                    break
                    
                edges = _random_3_reg_graph(n_try)
                if edges is None:
                    continue
                if _has_girth_at_least(n_try, edges, g):
                    best_n = n_try
                    best_edges = edges
                    break
    
    return best_n, best_edges

def _construct_luw(k, g, deadline=None):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    An explicit q-regular bipartite graph whose girth is at least d + 5 for odd d.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule table
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
        if deadline is not None and time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _random_3_reg_graph(n):
    """Generate a random simple 3-regular graph of n vertices. Returns None if impossible."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model with rejection
    for _ in range(10):  # Try up to 10 times
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        edges = set()
        valid = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if valid:
            return list(edges)
    return None

def _has_girth_at_least(n, edges, g):
    """Check if graph has no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
# EVOLVE-BLOCK-END
