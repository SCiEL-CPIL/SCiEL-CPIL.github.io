"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Try Greedy Random Construction for increasing n.
    2. Fallback to LZW Algebraic Construction.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Try greedy construction for smaller n first
    # The Moore bound for k=5, g=9 is 426.
    # Best known is 1116.
    # We try to find something smaller than the LZW fallback (1562).
    target_ns = [1100, 1200, 1300, 1400, 1500]
    
    for n in target_ns:
        if time.time() > deadline:
            break
        result = _greedy_construct(k, g, n, deadline)
        if result is not None:
            return result
            
    # Fallback to LZW
    return _lzw_construct(k, g)

def _greedy_construct(k, g, n, deadline):
    """
    Greedy construction of a k-regular graph with girth >= g.
    """
    # We will try a few times for each n
    max_restarts = 5
    for _ in range(max_restarts):
        if time.time() > deadline:
            return None
        
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # List of all potential edges
        # We don't generate all n^2/2 edges upfront to save memory/time, 
        # we iterate and check validity.
        
        edges_added = 0
        target_edges = n * k // 2
        
        # Shuffle vertices to break symmetry?
        # We maintain a list of vertices that still need edges
        active_vertices = list(range(n))
        random.shuffle(active_vertices)
        
        while edges_added < target_edges:
            if time.time() > deadline:
                return None
            
            # Find a valid edge
            found_edge = False
            # Try to pick an edge involving a vertex with high degree deficit
            # Sort active vertices by current degree (ascending) to prioritize those with fewer options?
            # Or just random.
            
            # Optimization: Pre-compute "forbidden" neighbors for each vertex?
            # Too expensive to update.
            
            # Randomly pick a vertex u that needs an edge
            # Filter active_vertices to those with deg < k
            candidates_u = [u for u in active_vertices if deg[u] < k]
            if not candidates_u:
                break # Should not happen if graph is regular and we stop at target_edges
            
            # Prioritize vertices with fewer available neighbors
            # Estimate available neighbors: n - 1 - deg[u] - (neighbors in adj[u])
            # This is O(n) per check.
            
            # Simple strategy: Random pick u
            u = random.choice(candidates_u)
            
            # Find valid v
            # v must be:
            # 1. v != u
            # 2. deg[v] < k
            # 3. v not in adj[u]
            # 4. dist(u, v) >= g in current graph (or adding edge doesn't create cycle < g)
            
            # Get all potential v
            potential_vs = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
            
            if not potential_vs:
                # Stuck. Restart or fail.
                return None
            
            # Filter potential_vs by girth constraint
            # We need to check if adding (u, v) creates a cycle < g.
            # This is equivalent to: is there a path from u to v of length < g-1?
            # If yes, adding (u,v) creates a cycle of length path_len + 1 < g.
            
            valid_vs = []
            for v in potential_vs:
                if _check_girth_constraint(adj, u, v, g):
                    valid_vs.append(v)
            
            if not valid_vs:
                # Stuck.
                return None
            
            # Pick v
            # Heuristic: Pick v that is "far" or random
            v = random.choice(valid_vs)
            
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            edges_added += 1
            
            # Update active_vertices? Not strictly necessary, we check deg in loop.
            # But removing satisfied vertices speeds up candidates_u.
            # We can just let the list comprehension handle it.
            
        # Check if all degrees are k
        if all(d == k for d in deg):
            # Final verification
            if _verify_girth(n, edges, g):
                return (n, edges)
                
    return None

def _check_girth_constraint(adj, u, v, g):
    """
    Check if adding edge (u, v) creates a cycle of length < g.
    Equivalent to: Is there a path from u to v of length <= g-2?
    If dist(u, v) <= g-2, then cycle length <= g-1.
    We require dist(u, v) >= g-1.
    """
    # BFS from u, stop at depth g-2
    # If we reach v within depth g-2, return False (constraint violated)
    # If we don't reach v, or reach it at depth >= g-1, return True
    
    if u == v:
        return False
        
    dist = {u: 0}
    queue = deque([(u, 0)])
    
    max_depth = g - 2
    
    while queue:
        node, d = queue.popleft()
        if d > max_depth:
            continue
        for nb in adj[node]:
            if nb == v:
                # Found v at distance d+1
                # Cycle length = d + 1 + 1 = d + 2
                # We need cycle length >= g
                # So we need d + 2 >= g  => d >= g - 2
                # If d < g - 2, then cycle < g.
                if d + 2 < g:
                    return False
                else:
                    return True
            if nb not in dist:
                dist[nb] = d + 1
                if d + 1 <= max_depth:
                    queue.append((nb, d + 1))
                    
    # If v not reached within max_depth, distance is > max_depth = g-2
    # So dist >= g-1.
    # Cycle length >= g.
    return True

def _verify_girth(n, edges, g):
    """
    Verify the graph has girth >= g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k: # k is not passed, use len check or pass k
            # We assume regularity from construction, but double check
            pass
            
    # BFS from each vertex to find shortest cycle
    # This is O(n * E)
    # For n=1200, E=3000. 1200 * 3000 = 3.6M ops. Fast.
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
                            
    return min_girth >= g

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
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

# EVOLVE-BLOCK-END
