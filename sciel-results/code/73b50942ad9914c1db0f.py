import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Estimate a starting n
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        r = g // 2 - 1
        moore = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # For k=5, g=9, moore is 106. Best known is 1116.
    # Random search works better when n is larger.
    # Let's start from a multiple of moore bound or a fixed offset.
    # For k=3, g=9, moore is 40, best is 70.
    # For k=5, g=9, let's try starting from 2 * moore and going up.
    
    n_current = max(moore + 10, int(moore * 1.5))
    
    best_n = float('inf')
    best_edges = None
    
    # Fallback: LUW construction from seed
    def luw_construct(k, g):
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

    # Get LUW result as a baseline
    try:
        n_luw, edges_luw = luw_construct(k, g)
        best_n = n_luw
        best_edges = edges_luw
    except Exception:
        pass

    if k < 3:
        return best_n, best_edges if best_edges else []

    # Local Search Strategy
    # We try to find a graph with girth >= g.
    # We start with n_current and increase if we fail.
    
    max_n = int(moore * 20) # Cap to avoid infinite loops
    
    while n_current <= max_n and time.time() < deadline - 10.0:
        
        # Try to solve for n_current
        success = False
        edges_found = None
        
        # Number of attempts
        attempts = 50
        
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Generate random k-regular graph
            adj, edges = generate_random_regular(k, n_current)
            if adj is None:
                continue
                
            # Local Search
            adj, edges = local_search_girth(k, g, adj, edges, time_limit=deadline - 5.0)
            
            # Check girth
            if check_girth(adj, g):
                success = True
                edges_found = edges
                break
        
        if success:
            if n_current < best_n:
                best_n = n_current
                best_edges = edges_found
            # If we found one, we can try smaller n? 
            # No, we started from a lower bound and went up. 
            # If we found one at n_current, it's the smallest in this sequence.
            # We can return immediately if we are confident.
            # But maybe a smaller n exists? We started from moore*1.5.
            # Let's return this one as it's likely close to optimal for random search.
            return n_current, edges_found
        
        # If failed, increase n
        n_current += max(10, int(n_current * 0.1))

    # If local search failed, return LUW
    return best_n, best_edges if best_edges else []

def generate_random_regular(k, n):
    """Generates a random k-regular simple graph using configuration model."""
    if k * n % 2 != 0:
        return None, None
    if k >= n:
        return None, None
        
    # Create stubs
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair stubs
    edges_set = set()
    adj = np.zeros((n, n), dtype=bool)
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, retry or swap
            # Simple retry
            return generate_random_regular(k, n)
        if adj[u, v]:
            # Multi-edge, retry
            return generate_random_regular(k, n)
            
        adj[u, v] = True
        adj[v, u] = True
        edges_set.add((min(u, v), max(u, v)))
        
    edges = list(edges_set)
    return adj, edges

def local_search_girth(k, g, adj, edges, time_limit):
    """
    Attempts to increase girth by swapping edges.
    Returns adj, edges.
    """
    # Pre-compute short cycles count? 
    # Too expensive. Just do random swaps and check girth periodically.
    
    # Heuristic: Just do many random valid swaps.
    # If girth is already >= g, we stop.
    
    # Check initial girth
    if check_girth(adj, g):
        return adj, edges
        
    # We need a metric to guide swaps. 
    # Count of cycles of length < g.
    # Let's just do random 2-swaps for a while and check girth.
    
    # To make it faster, we can track "bad" edges.
    # But for simplicity, let's do random swaps and check girth every 100 swaps.
    
    # Actually, checking girth is O(n * (k-1)^{g/2}).
    # For k=5, g=9, this is 5^4 = 625 per node. n=1000 -> 625k ops.
    # 100 swaps -> 62.5M ops. Might be too slow in Python.
    
    # Optimized Girth Check using NumPy?
    # BFS is better.
    
    # Let's limit the search time.
    start = time.time()
    max_swaps = 5000
    
    for _ in range(max_swaps):
        if time.time() > time_limit:
            break
            
        # Pick two random edges
        if len(edges) < 2:
            break
            
        e1_idx = random.randint(0, len(edges)-1)
        e2_idx = random.randint(0, len(edges)-1)
        if e1_idx == e2_idx:
            continue
            
        u, v = edges[e1_idx]
        x, y = edges[e2_idx]
        
        # Try swap 1: (u,x) and (v,y)
        if not adj[u, x] and not adj[v, y] and u != x and v != y and (u,x) not in [(min(a,b),max(a,b)) for a,b in [(u,v),(x,y)]]:
             # Check if (u,x) or (v,y) already exist
             if not adj[u, x] and not adj[v, y]:
                 # Perform swap
                 adj[u, v] = False
                 adj[v, u] = False
                 adj[x, y] = False
                 adj[y, x] = False
                 
                 adj[u, x] = True
                 adj[x, u] = True
                 adj[v, y] = True
                 adj[y, v] = True
                 
                 edges[e1_idx] = (min(u, x), max(u, x))
                 edges[e2_idx] = (min(v, y), max(v, y))
                 
                 # Check girth every 50 swaps
                 if _ % 50 == 0 and check_girth(adj, g):
                     return adj, edges
                 
        # Try swap 2: (u,y) and (v,x)
        elif not adj[u, y] and not adj[v, x] and u != y and v != x:
             if not adj[u, y] and not adj[v, x]:
                 adj[u, v] = False
                 adj[v, u] = False
                 adj[x, y] = False
                 adj[y, x] = False
                 
                 adj[u, y] = True
                 adj[y, u] = True
                 adj[v, x] = True
                 adj[x, v] = True
                 
                 edges[e1_idx] = (min(u, y), max(u, y))
                 edges[e2_idx] = (min(v, x), max(v, x))
                 
                 if _ % 50 == 0 and check_girth(adj, g):
                     return adj, edges

    # Final check
    if check_girth(adj, g):
        return adj, edges
        
    return adj, edges

def check_girth(adj, g):
    """
    Returns True if girth(adj) >= g.
    Uses BFS from each vertex.
    """
    n = adj.shape[0]
    # We need to check for cycles of length < g.
    # A cycle of length L exists if there is a path of length L-1 from u to v where v is a neighbor of u, 
    # and the path doesn't use the edge (u,v) directly? 
    # Simpler: BFS from each vertex, track distance and parent.
    # If we find a neighbor at distance < g-1, we have a cycle of length dist+1.
    
    # Optimization: Only need to check up to depth R = g//2 - 1?
    # No, if g is odd, say 9, cycles of length 8, 7, 6, 5, 4, 3.
    # Max cycle length to check is g-1.
    # BFS depth should be (g-1)//2.
    # If we find a connection at depth d, cycle length is 2d+1 (odd) or 2d (even).
    # Actually, standard girth check:
    # For each vertex s, BFS.
    # If we visit a vertex v that is already visited by s, and dist[s][v] < g, and the edge (s,v) is not the one used to reach v?
    # No, simpler:
    # For each vertex s:
    #   BFS.
    #   If we find a back-edge to an ancestor in the BFS tree, calculate cycle length.
    #   If cycle length < g, return False.
    
    # Even simpler for small g:
    # Check for cycles of length 3, 4, ..., g-1.
    
    # Let's use a standard BFS girth check.
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u, v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < g: # Only care if dist < g
                            queue.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But this counts the edge (u,v) twice if it's the tree edge?
                        # No, if (u,v) is a tree edge, we wouldn't be in this branch?
                        # Actually, if v is visited, and (u,v) is not the edge used to visit v from parent[v],
                        # then we have a cycle.
                        # However, in an undirected graph, every non-tree edge creates a cycle.
                        # We must ensure we don't count the parent edge.
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
                                
    return True
