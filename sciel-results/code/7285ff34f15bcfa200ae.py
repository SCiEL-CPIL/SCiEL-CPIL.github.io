import time
import random
from math import factorial
from collections import deque

DEADLINE = 112  # seconds; 8s safety margin under 120s limit

def _girth_at_least_g(n, adj, g):
    """Check if girth of graph is at least g using non-backtracking BFS.
    adj: list of lists of neighbors.
    Returns True if girth >= g, False otherwise.
    """
    for s in range(n):
        # BFS from s with non-backtracking
        # State: (vertex, parent)
        # We track dist[(v, p)] but that's expensive.
        # Instead: standard BFS, and when we find a back edge (u,v) where v is visited
        # and v != parent[u], cycle length = dist[u] + 1 + dist[v].
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque()
        queue.append(s)
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= g:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    if d + 1 < g:
                        queue.append(v)
                else:
                    if dist[v] < d + 1:
                        # Cycle found
                        cycle_len = d + 1 + dist[v]
                        if cycle_len < g:
                            return False
    return True

def construct(k, g):
    t0 = time.time()
    rng = random.Random(12345)
    
    best_n = None
    best_edges = None
    
    # Try Cayley graphs over S_m
    # For k=3, g=16: try m=8 (n=40320) first
    # Then try m=7 (n=5040) for smaller graphs
    
    for m in [8, 7, 9, 6]:
        if time.time() - t0 > DEADLINE - 15:
            break
        n = factorial(m)
        if n > 100000:
            continue
        
        identity = tuple(range(m))
        
        def compose(p1, p2):
            return tuple(p1[p2[i]] for i in range(m))
        
        def is_involution(p):
            return compose(p, p) == identity
        
        # Generate all permutations
        perms = [identity]
        for i in range(1, m):
            new_perms = []
            for p in perms:
                p_list = list(p)
                for pos in range(i + 1):
                    new_p = p_list[:pos] + [i] + p_list[pos:]
                    new_perms.append(tuple(new_p))
            perms = new_perms
        
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        
        # Try random generator sets
        max_tries = 30
        for trial in range(max_tries):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Pick k distinct involutions (for undirected Cayley graph)
            gens = []
            attempts = 0
            while len(gens) < k and attempts < 300:
                p = list(range(m))
                rng.shuffle(p)
                p = tuple(p)
                if p == identity:
                    continue
                if is_involution(p) and p not in gens:
                    gens.append(p)
                attempts += 1
            
            if len(gens) < k:
                continue
            
            # Build adjacency
            adj = [None] * n
            for gi, g_elem in enumerate(perms):
                neighbors = []
                for s in gens:
                    ng = compose(s, g_elem)
                    neighbors.append(perm_to_idx[ng])
                adj[gi] = neighbors
            
            # Check for self-loops
            if any(u in adj[u] for u in range(n)):
                continue
            
            # Check for multi-edges (duplicate neighbors)
            valid = True
            for u in range(n):
                if len(set(adj[u])) < k:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check girth >= g
            if time.time() - t0 > DEADLINE - 3:
                break
            
            if _girth_at_least_g(n, adj, g):
                # Build edge list
                edge_set = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_set.add((u, v))
                        elif u > v:
                            edge_set.add((v, u))
                edges = list(edge_set)
                
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid graph for this m
        
        if best_n is not None:
            # Continue to try smaller m for potentially smaller graphs
            # But m=7 (5040) < m=8 (40320), so we want to try m=7 after m=8
            # The loop already goes 8,7,9,6 so it will try m=7 next
            pass
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
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
