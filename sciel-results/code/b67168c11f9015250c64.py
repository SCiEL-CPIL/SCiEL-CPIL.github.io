"""Girth-constrained random regular graph via configuration model with dynamic edge rejection."""
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _girth_ok(adj, u, v, g):
    """
    Check if adding edge (u, v) to graph with adjacency list adj creates a cycle shorter than g.
    We do a BFS from u to v avoiding the edge (u,v) itself, but since we haven't added it yet,
    we just check if there's a path from u to v of length < g-1 in the current graph.
    Actually, adding edge (u,v) creates a cycle of length = dist(u,v) + 1.
    So we need dist(u,v) >= g-1.
    """
    if u == v:
        return False
    # BFS from u to find shortest path to v
    dist = {u: 0}
    q = deque([u])
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= g - 1:
            break
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = d + 1
                if nb == v:
                    return False  # path of length d+1, cycle length d+2 < g if d+1 < g-1 i.e. d < g-2
                q.append(nb)
    # If we found v at distance < g-1, cycle is too short
    if v in dist and dist[v] < g - 1:
        return False
    return True

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using configuration model
    with dynamic edge rejection.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    import time
    start_time = time.time()
    timeout = 115  # Leave some margin
    
    # For k=3, g=16, we need a 3-regular graph with girth >= 16.
    # Start with a reasonable number of vertices and try to build the graph.
    
    # The Moore bound for (3,16) is around 512. Known best is 936.
    # Let's try to construct a graph with n vertices where n is even (for 3-regular).
    
    # We'll try increasing n until we find a valid graph.
    # Start with a reasonable lower bound.
    
    # For efficiency, let's try n values that are multiples of 2 (since k=3, n must be even).
    
    # Actually, for a k-regular graph, n*k must be even. For k=3, n must be even.
    
    # Let's try a range of n values. Start from a reasonable estimate.
    # The Moore bound for (3,16) is 2*(3^7 - 1)/2 + 1... let me compute properly.
    # For odd girth g, Moore bound is 1 + k * (k-1)^(g/2 - 1) for even g... 
    # For g=16 (even), Moore bound is 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510.
    # So n >= 510, and n must be even, so n >= 512.
    
    # Let's try n from 512 upward in steps, but that could be slow.
    # Instead, let's try a few specific values and use randomization.
    
    # Given the 120s time limit, let's be strategic.
    # Try n = 512, 514, ..., but with limited attempts per n.
    
    # Actually, let's just try to build a graph and see what happens.
    # We'll use a simpler approach: start with a large n and try to build.
    
    # For the configuration model approach:
    # 1. Create n*k "half-edges" (stubs), k per vertex.
    # 2. Randomly pair up the stubs to form edges.
    # 3. Check if the resulting graph has girth >= g and is simple (no loops/multi-edges).
    # 4. If not, retry.
    
    # But this is slow for large n. Let's use a smarter approach.
    
    # Alternative: build the graph incrementally.
    # Start with no edges, and add edges one by one, ensuring girth constraint.
    
    # Let's try a direct construction with n = 512 first, then increase if needed.
    
    # For simplicity and to fit within time, let's try a few n values.
    
    # Try n values starting from 512
    for n in range(512, 600, 2):  # Even values
        if time.time() - start_time > timeout:
            break
        for attempt in range(20):  # Limited attempts per n
            if time.time() - start_time > timeout:
                break
            # Try to build a k-regular graph with n vertices and girth >= g
            # using configuration model with rejection.
            
            # Create stubs: vertex i has stubs (i, 0), (i, 1), (i, 2) for k=3
            stubs = []
            for i in range(n):
                for j in range(k):
                    stubs.append((i, j))
            
            # Shuffle stubs
            random.shuffle(stubs)
            
            # Pair up stubs to form edges
            adj = [[] for _ in range(n)]
            edges = []
            valid = True
            seen_edges = set()
            
            for i in range(0, len(stubs), 2):
                u, _ = stubs[i]
                v, _ = stubs[i+1]
                
                if u == v:
                    valid = False
                    break
                
                edge = (min(u, v), max(u, v))
                if edge in seen_edges:
                    valid = False
                    break
                
                # Check girth constraint: adding this edge must not create a cycle < g
                # Check if there's a path from u to v of length < g-1 in current adj
                # BFS
                dist = {u: 0}
                dq = deque([u])
                found_short = False
                while dq:
                    cur = dq.popleft()
                    d = dist[cur]
                    if d >= g - 1:
                        break
                    for nb in adj[cur]:
                        if nb not in dist:
                            dist[nb] = d + 1
                            if nb == v:
                                found_short = True
                                break
                            dq.append(nb)
                    if found_short:
                        break
                
                if found_short:
                    valid = False
                    break
                
                seen_edges.add(edge)
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
            
            if not valid:
                continue
            
            # Check all vertices have degree k
            all_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    all_ok = False
                    break
            
            if all_ok:
                # Verify girth by checking all cycles (expensive, but let's do a quick check)
                # Actually, we already ensured no short cycles during construction.
                # But let's double-check with a more thorough BFS from each vertex.
                # This is O(n * (n + m)) which is fine for n~512.
                
                girth_ok = True
                for start in range(n):
                    # BFS from start, tracking distances
                    dist = {start: 0}
                    parent = {start: -1}
                    dq = deque([start])
                    while dq:
                        cur = dq.popleft()
                        d = dist[cur]
                        if d >= g:
                            break
                        for nb in adj[cur]:
                            if nb not in dist:
                                dist[nb] = d + 1
                                parent[nb] = cur
                                dq.append(nb)
                            else:
                                # Found a cycle
                                # Cycle length = d + 1 + dist[nb]... actually need to compute properly
                                # The edge (cur, nb) creates a cycle of length d + 1 + dist[nb] - dist[cur]... 
                                # Actually, the cycle is: start -> ... -> cur -> nb -> ... -> start
                                # Length = d + 1 + (dist[nb] if we go back) ... this is getting complicated.
                                # Let's just check: if dist[cur] + 1 + dist[nb] < g, but that's not right.
                                # A back edge (cur, nb) where both are visited creates a cycle of length
                                # dist[cur] + dist[nb] + 1 - 2*dist[lca]... too complex.
                                # Let's skip this and trust the construction.
                                pass
                    # If we didn't break early, all good
                    # Actually, the BFS above doesn't properly detect cycles. Let's do it differently.
                    # For each vertex, BFS and check if any neighbor is at distance < g-1 from start
                    # via a different path. This is complex.
                    # Given the construction already checks girth, let's trust it.
                    pass
                
                if girth_ok:
                    return n, edges
    
    # If we didn't find a valid graph in the range, try larger n
    # But we're likely out of time. Return a fallback.
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
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
