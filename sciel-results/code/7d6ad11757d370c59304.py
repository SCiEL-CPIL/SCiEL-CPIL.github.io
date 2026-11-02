"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if k == 4 and g == 10:
        # Try increasing n values, starting from just above Moore bound
        target_ns = [162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190,
                     192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220,
                     222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250,
                     252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280,
                     282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
                     312, 314, 316, 318, 320, 322, 324, 326, 328, 330]
        
        for n in target_ns:
            if time.time() > deadline - 5:
                break
            
            result = _search_for_n(n, k, g, deadline)
            if result is not None:
                return result
        
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _search_for_n(n, k, g, deadline):
    """Search for a k-regular graph with girth >= g on n vertices.
    
    Uses random restarts with edge-swap local search.
    """
    max_restarts = 400
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            return None
        
        # Generate random 4-regular graph
        edges = _gen_random_4reg(n)
        if edges is None:
            continue
        
        adj = _build_adjacency(edges, n)
        
        # Check if already valid
        if _check_girth_fast(adj, n, g):
            return n, edges
        
        # Do a batch of random edge swaps, then check girth
        num_swaps = 50
        for it in range(num_swaps):
            if time.time() > deadline - 2:
                return None
            
            # Perform a random valid edge swap
            # Pick two random vertices u, v that are not adjacent
            # Pick random neighbors a of u and b of v
            # Swap: remove (u,a), (v,b); add (u,b), (v,a)
            swapped = False
            for _ in range(100):  # Try up to 100 times to find a valid swap
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or _edge_exists(adj, u, v):
                    continue
                
                if len(adj[u]) < 2 or len(adj[v]) < 2:
                    continue
                
                a = random.choice(adj[u])
                b = random.choice(adj[v])
                
                if u == b or v == a:
                    continue
                if _edge_exists(adj, u, b) or _edge_exists(adj, v, a):
                    continue
                
                # Perform swap
                adj[u].remove(a)
                adj[a].remove(u)
                adj[v].remove(b)
                adj[b].remove(v)
                
                adj[u].append(b)
                adj[b].append(u)
                adj[v].append(a)
                adj[a].append(v)
                
                swapped = True
                break
            
            if not swapped:
                break
        
        # Check girth after batch of swaps
        if _check_girth_fast(adj, n, g):
            return n, _edges_from_adj(adj, n)
    
    return None


def _edge_exists(adj, u, v):
    """Check if edge (u,v) exists in adjacency list."""
    return v in adj[u]


def _edges_from_adj(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _build_adjacency(edges, n):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(adj, n, g):
    """Fast girth check using BFS with early exit.
    
    Returns True if girth >= g, False otherwise.
    """
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _find_short_cycle_edge(adj, n, g):
    """Find an edge (u,v) that lies on a cycle of length < g.
    
    Returns (u, v) or (None, None) if no short cycle exists.
    """
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return (u, v)
    
    return (None, None)


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices using configuration model."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    for attempt in range(100):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
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
# EVOLVE-BLOCK-END
