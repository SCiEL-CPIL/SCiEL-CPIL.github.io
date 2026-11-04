import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # Fallback LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def check_girth_bfs(adj, n, g):
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                nd = dist[node] + 1
                if nd >= g:
                    break
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = nd
                        parent[nb] = node
                        queue.append(nb)
        return min_cycle

    # Specific handling for k=3, g=16
    if k == 3 and g == 16:
        # Moore bound for k=3, g=16 is 510.
        # Voltage graph over K4 has 4*q vertices.
        # We need 4*q >= 510 => q >= 127.5 => q >= 128.
        
        # Precompute closed walks in K4 of length < 16
        # K4 vertices: 0,1,2,3
        # Edges: all pairs
        base_vertices = [0, 1, 2, 3]
        base_adj = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
        
        # Generate walks
        # A walk is a sequence of vertices v0...vL with v0=vL
        # We represent a walk by the sequence of edges (or vertices)
        # For voltage check, we need the sequence of directed edges.
        # Let's generate all closed walks of length L up to 15.
        # To avoid duplicates and backtracks in the "simple cycle" sense, 
        # we actually need ALL closed walks because non-simple walks can form simple cycles in the lift.
        
        walks = []
        # Start from each vertex
        for start_v in base_vertices:
            # DFS to find closed walks
            # State: current_v, path (list of vertices), length
            # We want length < 16
            def dfs_walks(current_v, path, length):
                if length == 0:
                    if current_v == start_v and len(path) > 0:
                        # Found a closed walk
                        # Store the sequence of edges (directed)
                        # path is v0, v1, ..., vL
                        # Edges are (v0,v1), (v1,v2), ...
                        edges_seq = []
                        for i in range(len(path)-1):
                            edges_seq.append((path[i], path[i+1]))
                        walks.append(edges_seq)
                    return
                
                for next_v in base_adj[current_v]:
                    # Avoid immediate backtrack? 
                    # No, we need all walks.
                    # But to reduce duplicates, we can enforce a canonical form?
                    # Actually, for voltage check, direction matters.
                    # (u,v) and (v,u) are different edges in the voltage sense.
                    dfs_walks(next_v, path + [next_v], length + 1)
            
            dfs_walks(start_v, [start_v], 0)
        
        # Remove duplicates (same edge sequence)
        unique_walks = list(set([tuple(w) for w in walks]))
        
        # Map edge to index for voltage array
        # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        edge_map = {
            (0,1): 0, (1,0): 0,
            (0,2): 1, (2,0): 1,
            (0,3): 2, (3,0): 2,
            (1,2): 3, (2,1): 3,
            (1,3): 4, (3,1): 4,
            (2,3): 5, (3,2): 5
        }
        
        # For each walk, precompute the indices of the edges
        # And the sign (direction)
        # Voltage of (u,v) is g[u,v].
        # If walk uses (u,v), add g[u,v].
        # If walk uses (v,u), add g[v,u] = -g[u,v].
        
        walk_constraints = []
        for w in unique_walks:
            coeffs = [0] * 6
            for u, v in w:
                idx = edge_map[(u,v)]
                # Determine sign
                # We define g[u,v] as the voltage from u to v.
                # In our edge_map, (1,0) maps to 0, same as (0,1).
                # We need to know if we are going "forward" or "backward" relative to a canonical direction.
                # Let's define canonical direction: u < v.
                if u < v:
                    coeffs[idx] += 1
                else:
                    coeffs[idx] -= 1
            
            # Skip if all coeffs are 0 (should not happen for closed walks)
            if any(c != 0 for c in coeffs):
                walk_constraints.append(coeffs)
        
        # Search for q
        rng = random.Random(42)
        q_start = 128
        max_q = 300
        
        for q in range(q_start, max_q + 1):
            if time.time() > deadline - 10:
                break
            
            # Try to find voltages for this q
            found = False
            # Number of trials
            # Probability of success is decent for q > 128
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random voltages
                voltages = [rng.randint(0, q-1) for _ in range(6)]
                
                # Check constraints
                valid = True
                for coeffs in walk_constraints:
                    s = 0
                    for i in range(6):
                        s += coeffs[i] * voltages[i]
                    if s % q == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct the graph
                    n = 4 * q
                    adj = [[] for _ in range(n)]
                    edge_set = set()
                    
                    # Build edges
                    for u, v in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
                        idx = edge_map[(u,v)]
                        vu = voltages[idx]
                        # Voltage from u to v is vu
                        # Voltage from v to u is -vu
                        for fiber in range(q):
                            su = u * q + fiber
                            sv = v * q + (fiber + vu) % q
                            if su < sv:
                                edge_set.add((su, sv))
                            else:
                                edge_set.add((sv, su))
                    
                    for su, sv in edge_set:
                        adj[su].append(sv)
                        adj[sv].append(su)
                    
                    # Verify girth
                    min_c = check_girth_bfs(adj, n, g)
                    if min_c >= g:
                        edges = list(edge_set)
                        return n, edges
                    else:
                        # If BFS says invalid, our constraint check missed something
                        # (e.g. non-simple walk forming simple cycle)
                        # Continue searching
                        pass
                
        # If no small q found, fallback to LUW
        return luw_construct(k, g)
    
    # General fallback
    return luw_construct(k, g)
