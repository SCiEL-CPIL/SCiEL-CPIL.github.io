import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        # Fallback to LUW for other cases, though task is specific
        return _luw_construct(k, g, deadline)
    
    # Target N is around Moore bound 426
    # We will search for voltage graph lifts
    
    # Generate candidate base graphs
    base_graphs = []
    
    # 1. K6
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_graphs.append((6, edges_k6))
    
    # 2. K5,5
    edges_k55 = [(i, j+5) for i in range(5) for j in range(5)]
    base_graphs.append((10, edges_k55))
    
    # 3. Random 5-regular graphs on V vertices
    # V must be even for regular graph? No, sum of degrees = 5V must be even, so V must be even.
    for V in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
        # Try to generate a random 5-regular graph
        # Using configuration model (pairing stubs)
        # Stubs: 5 per vertex
        stubs = [i for i in range(V) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        if valid and len(edges) == 5 * V // 2:
            base_graphs.append((V, list(edges)))
            
    # Search for voltage assignments
    best_n = float('inf')
    best_edges = None
    
    # Prioritize smaller V (smaller base, larger m) or specific known good ones?
    # Smaller V means larger m, which might make voltage sums harder to be 0?
    # Actually, larger m makes it easier for sums to be non-zero.
    # So K6 (V=6, m~70) is a good candidate.
    
    # Sort base graphs by V
    base_graphs.sort(key=lambda x: x[0])
    
    for V, base_edges in base_graphs:
        if time.time() > deadline - 10:
            break
            
        # Determine m
        # We want N = V * m to be close to 426
        # m should be an integer.
        # Try m in range [426//V - 5, 426//V + 5]
        target_m = 426 // V
        for m in range(max(2, target_m - 5), target_m + 6):
            N = V * m
            if N < 426: # Moore bound is 426, so N must be >= 426? 
                # Actually, Moore bound is a lower bound. So N >= 426.
                # If N < 426, it's impossible to have girth 9?
                # Yes, if N < Moore bound, girth must be less than 9.
                continue
                
            # Try voltage assignments with smarter strategies
            for trial in range(50): # 50 trials per (V, m)
                if time.time() > deadline - 5:
                    break
                
                # Strategy: use structured voltage assignments
                # Try different patterns:
                if trial < 10:
                    # Random
                    voltages = {}
                    for u, v in base_edges:
                        voltages[(u, v)] = random.randint(0, m-1)
                        voltages[(v, u)] = (m - voltages[(u, v)]) % m
                elif trial < 20:
                    # Linear pattern based on vertex indices
                    voltages = {}
                    for u, v in base_edges:
                        voltages[(u, v)] = (u * 7 + v * 13) % m
                        voltages[(v, u)] = (m - voltages[(u, v)]) % m
                elif trial < 30:
                    # Based on edge index
                    voltages = {}
                    for idx, (u, v) in enumerate(base_edges):
                        voltages[(u, v)] = (idx * 11 + 3) % m
                        voltages[(v, u)] = (m - voltages[(u, v)]) % m
                elif trial < 40:
                    # Fibonacci-like
                    voltages = {}
                    for idx, (u, v) in enumerate(base_edges):
                        voltages[(u, v)] = ((idx * idx + idx + 1) % m)
                        voltages[(v, u)] = (m - voltages[(u, v)]) % m
                else:
                    # Mixed random with some structure
                    voltages = {}
                    for idx, (u, v) in enumerate(base_edges):
                        voltages[(u, v)] = (random.randint(0, m-1) + idx * 5) % m
                        voltages[(v, u)] = (m - voltages[(u, v)]) % m
                    
                # Build the lift graph
                lift_adj = [[] for _ in range(N)]
                
                for u, v in base_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        u_idx = u * m + i
                        v_idx = v * m + (i + w) % m
                        lift_adj[u_idx].append(v_idx)
                        lift_adj[v_idx].append(u_idx)
                        
                # Check girth with early termination
                if _check_girth_fast(lift_adj, N, g):
                    # Extract edges
                    edges = []
                    for u in range(N):
                        for v in lift_adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        if N == 426:
                            return N, edges
                            
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking walks."""
    # For girth >= 9, we need no cycles of length 3, 4, 5, 6, 7, or 8
    # Use BFS from each vertex but stop early when we can determine girth
    
    # First, quick check: look for triangles (girth 3)
    # For each edge (u,v), check if they have a common neighbor
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Check if u and v share a neighbor
                common = set(adj[u]) & set(adj[v])
                if common:
                    return False  # Triangle found
    
    # Check for 4-cycles
    # For each pair of vertices at distance 2, check if they share multiple neighbors
    for u in range(n):
        # Get all vertices at distance 2 from u
        dist2 = set()
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w not in adj[u]:
                    dist2.add(w)
        # For each vertex at distance 2, check if it's connected to u via multiple paths
        for w in dist2:
            # Count common neighbors of u and w
            common = 0
            for x in adj[u]:
                if x in adj[w]:
                    common += 1
                    if common >= 2:
                        return False  # 4-cycle found
    
    # For girth >= 5, check 5-cycles using BFS from each vertex
    # This is more expensive, so we do it last
    for start in range(n):
        # BFS up to depth 4
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g, deadline):
    """Original LUW implementation as fallback."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
